# Bristle

**The Universal HID Aggregation and Normalization Layer**

Bristle is the structural spine where every poke, press, scroll, and wiggle gets combed into a clean stream. It speaks *many* device dialects (PS/2 scancodes, USB HID reports, Bluetooth HID, gamepads, touchpads) and emits *one* dialect: **Bristle events** (typed, timestamped, stateful).

> **Naming Note:** *Thigmonasty* remains the cute behavioral name for "touch causes reaction" — the keyboard-state engine (pressed set, repeat, mods) that lives *inside* Bristle.

---

## Architecture

Bristle runs as a **userspace service** (like Blossom). The kernel provides device capability primitives; individual drivers run in userspace; Bristle is the sole "input authority" that apps subscribe to.

```
[ps2_kbd]     [usb_hid_kbd]     [bt_hid_kbd]     [ps2_mouse] ...
     │              │                │                 │
     └──── raw reports over Ports ───┴───────── raw reports ─┘
                            │
                         Bristle
            (normalize + state + routing + policy-lite)
                            │
       ┌───────────────┬───────────────┬────────────────┐
       │               │               │                │
   focused app     compositor       shell / REPL     (privileged)
  (receives keys) (receives all)   (receives hotkeys)
```

**Apps don't talk to drivers. Apps talk to Bristle.**

---

## Ownership

Bristle owns input state in one place:

- Modifiers and locks (`shift`, `ctrl`, `alt`, `meta`, `altgr`, `caps`, `num`, `scroll`)
- Pressed keys set
- Pointer buttons and position (relative/absolute)
- Wheel deltas
- Device identities

---

## Interfaces

### 1. Driver Input Interface (Drivers → Bristle)

Drivers send raw input reports in a driver-neutral envelope:

| Field | Description |
|-------|-------------|
| `device_id` | ThingId of the device node in the graph |
| `kind` | `keyboard` / `mouse` / `touch` / `gamepad` / `consumer` |
| `timestamp` | Monotonic nanoseconds |
| `payload` | Opaque bytes or small typed struct |

### 2. App Subscription Interface (Bristle → Apps)

Apps subscribe to an event stream with filtering:

- By seat (future)
- By device kind
- By focus routing (only focused app gets text/keys)
- By privileged channels (compositor receives all pointer movement)

---

## Event Model

### Core Event Types

```rust
KeyDown { key: Key, mods: Mods, repeat: bool }
KeyUp { key: Key, mods: Mods }
TextInput { utf8: [..] }  // Stage 2, from keymap service
PointerMove { dx, dy }    // Relative
PointerAbs { x, y }       // Absolute/touch
PointerButtonDown { button }
PointerButtonUp { button }
Scroll { dx, dy }
DeviceAdded { device_id, capabilities }
DeviceRemoved { device_id }
```

### Key Representation

Normalized `Key` enum along HID usage lines:

- `Key::A`, `Key::Enter`, `Key::F1`, `Key::LeftShift`, etc.
- Escape hatch: `Key::UsagePage { page, usage }` for exotic devices

### Repeat Policy

Bristle owns repeat (per-device but unified per seat). Repeat generates `KeyDown(repeat=true)`.

---

## Driver Integration

### PS/2 Keyboard

1. Claims i8042/PS2 controller via LPC path
2. Reads scancodes (set 1/2)
3. Emits `RawKeyboardReport` → Bristle translates scancode → `Key`

### USB HID Keyboard

1. USB stack yields HID input reports
2. Driver parses HID report descriptor (modifiers, key usages)
3. Emits `RawHidReport { usage_page, report_bytes }`
4. Bristle does delta computation (HID reports pressed-set, not edges)

### Bluetooth HID

- BT stack yields HID input reports
- Reuses same HID report parsing logic
- Same feed into Bristle

**Key idea:** Bristle doesn't care how the report arrived.

---

## Internal Modules

| Module | Responsibility |
|--------|----------------|
| **Ingress** | Per-driver port readers, decode envelopes |
| **Device Registry** | Map `device_id` → device state machine, store capabilities |
| **Normalizers** | `ps2_normalizer`, `hid_normalizer`, `mouse_normalizer` |
| **State** | Keyboard state (pressed set, mods, locks), pointer state |
| **Router** | Focus-based event routing, privileged channels |

---

## Security Model

### Rule of Least Authority

- Drivers have hardware access but not app ports
- Apps subscribe only via Bristle
- Bristle enforces focus and capability

### Keylogger Prevention

Require explicit capability to receive:

- Raw events from all devices
- Events when not focused
- Text input vs raw key events

**Defaults:**

- Focused app gets key events
- Compositor gets pointer events
- Only privileged shell/WM gets global shortcuts

---

## Graph Integration

### Nodes

- `svc.Input` (Bristle)
- `dev.hid.Keyboard`, `dev.hid.Mouse`, etc. (per physical device)
- `seat` (future)

### Edges

```cypher
(svc.Input)-[:CONSUMES]->(dev.hid.Keyboard)
(svc.Input)-[:ROUTES_TO]->(app.X)
(dev.hid.Keyboard)-[:BACKED_BY]->(dev.ps2.Controller)
```

---

## Roadmap

### Phase 0: Skeleton
- Bristle receives `RawKeyEdge` from PS/2 keyboard driver
- Maintains pressed/mod state
- Routes events to one subscribed app (`input_echo`)

### Phase 1: Mouse + Focus
- Add PS/2 mouse (pointer move + buttons)
- Add focus routing

### Phase 2: USB HID
- Plug in USB HID keyboard
- Implement HID delta computation (pressed-set diff)

### Phase 3: Text Input
- Add keymap/compose layer as separate service (i18n stays out of Bristle)

---

## Taxonomy

| Name | Role |
|------|------|
| **Bristle** | Input broker + normalizer + router |
| **Thigmonasty** | Keyboard-state engine inside Bristle (pressed set, repeat, mods) |
| **Input Echo** | Test app that subscribes to Bristle and prints events |

*Plant pun: Bristle = the "outer" tactile system; Thigmonasty = the "reflex arc" for keys.*
