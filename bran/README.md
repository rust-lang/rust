# Bran

**Bran** is the **Boot Runtime Abstraction** for ThingOS.

The name officially (and unofficially) stands for:

> **B**oot **R**esponsibility & **A**rchitecture **N**onsense
> or, if you must, **Boot Runtime Abstractio-N**

Bran exists to take all the *messy, architecture-specific, bootloader-specific nonsense* and package it into a **small, stable, boring interface** that the kernel proper can rely on.

If the kernel is *what* the system does,
Bran is *how the system wakes up*.

---

## What Bran Is

Bran is a **thin abstraction layer** that sits between:

* the **bootloader** (currently Limine)
* the **CPU architecture** (x86_64, aarch64, loongarch64, …)
* and the **kernel core** (`/kernel`)

Its job is to:

* Normalize bootloader quirks
* Hide architectural differences
* Provide early-boot facilities
* Present a clean, uniform runtime to the kernel

Bran is **not** policy.
Bran is **not** the kernel.
Bran is **not** userspace.

Bran is the **soil** the kernel grows out of.

---

## What Bran Handles

Bran is responsible for *everything that must happen before the kernel can think clearly*.

That includes:

### 🧠 Architecture Abstraction

* CPU feature discovery
* Architecture-specific initialization
* Register setup
* Early interrupt configuration
* Architecture quirks and workarounds

### 🥾 Bootloader Integration

* Limine request/response handling
* Module discovery
* Boot memory map parsing
* Framebuffer discovery
* Boot-time metadata extraction

The kernel should **never** talk to Limine directly.
Bran speaks Limine so the kernel doesn’t have to.

### 🧱 Early Memory Information

* Physical memory ranges
* Memory kinds (usable, reserved, firmware, etc.)
* Confidence / provenance tracking
* Early allocation helpers (if absolutely necessary)

Bran *describes* memory.
The kernel *manages* memory.

### 🖨️ Early Console & Logging

* Early serial output
* Early framebuffer output (if available)
* Panic-safe printing

Bran gives the kernel a voice *before* everything else is online.

### ⏱️ Early Time Sources

* Boot-time timestamps
* Architecture timers
* RTC access (where appropriate)

Bran exposes time primitives; the kernel decides what “time” means.

---

## What Bran Explicitly Does **Not** Do

Bran is intentionally constrained.

It does **not**:

* Schedule threads
* Allocate heaps
* Manage virtual memory long-term
* Expose syscalls
* Maintain global state beyond boot
* Implement policy decisions
* Know anything about the graph, root, sprout, bloom, or leaves

If something feels like it belongs in the kernel — it probably does.

If something feels like “this is gross but unavoidable at boot” — that’s Bran.

---

## The Kernel–Bran Contract

Bran exposes a **small, explicit interface** to the kernel.

Think of it as:

> “Here is the world as it exists *right now*.
> Good luck.”

The kernel is expected to:

* Copy or consume the data it needs
* Transition ownership cleanly
* Outgrow Bran as soon as possible

Bran is **not long-lived**.
It exists to get the kernel standing — then quietly steps back.

---

## Architecture Support

Bran is structured to support **multiple architectures cleanly**.

Each architecture provides:

* Its own boot/runtime implementation
* Its own Limine glue (if applicable)
* A shared interface exposed to the kernel

Example layout (conceptual):

```
bran/
├── src/
│   ├── lib.rs
│   ├── runtime.rs        # Architecture-agnostic interface
│   └── arch/
│       ├── x86_64/
│       ├── aarch64/
│       └── loongarch64/
```

The kernel never imports from `arch::*`.
It only talks to the runtime abstraction.

---

## Design Principles

Bran follows a few non-negotiable principles:

### 🪶 Thinness

Bran should be *as small as possible*.
Every line of code here is paid for forever.

### 🧼 Explicitness

No hidden state.
No magic globals.
No “just trust me” behavior.

### 🧱 Stability

The Bran → Kernel interface should change *slowly* and *deliberately*.

### 🌱 Replaceability

Bran should be easy to rewrite, replace, or delete.
That’s how we know it’s doing its job.

---

## Relationship to the Rest of ThingOS

Bran is part of the **boot lineage**:

```
Firmware → Bootloader → Bran → Kernel → Sprout → Root → Bloom → Leaves
```

Bran is the **last boot-specific layer**.

Everything above it should be able to pretend the machine simply *exists*.

---

## In Short

Bran is:

* The boot runtime abstraction
* The architecture containment zone
* The place where unavoidable nonsense goes to die

Or, more simply:

> **Bran lets the kernel be honest.**