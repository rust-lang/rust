# Filesystem Window Model

## Conceptual Overview

Janix exposes GUI objects as ordinary filesystem objects so Bloom can discover, inspect, and manage them through the same VFS model used everywhere else.

The design keeps the Wayland-friendly split intact:

- A `surface` is drawable content plus attach/damage/commit state.
- A `window` is shell policy and compositor-owned state.
- Configure and close travel from compositor to client through an event file.
- Attach and commit travel from client to compositor through per-surface endpoints.

This is intentionally Wayland-friendly, not Wayland-literal. The namespace mirrors the concepts that matter without reproducing the Wayland wire protocol as files.

## Namespace Layout

The kernel mounts `/session` as a writable tmpfs. Bloom expects two primary trees:

```text
/session/
  windows/
    <wid>/
      shell/
        role
        title
        app_id
        parent
        requested/
          maximize
          fullscreen
          minimize
        current/
          x
          y
          width
          height
          z
          activated
          maximized
          fullscreen
          resizing
      bind/
        surface
      events
      status/
        mapped
        focused
        last_configure_serial
        client_pid
        closing
  surfaces/
    <sid>/
      attach
      damage
      commit
      input_region
      opaque_region
      status/
        mapped
        last_commit
        configured_serial
        width
        height
        buffer_attached
```

`x`, `y`, and `z` live under `shell/current/` because placement and stacking are compositor-owned shell state in practice.

## Protocol Semantics

### Surface Attach / Commit

Clients write surface content through memfd-backed attachments.

`attach` currently uses a simple key/value payload:

```text
fd=42
width=800
height=600
stride=3200
format=1
```

The attached `fd` must be a memfd containing BGRA-style pixels. Bloom treats `commit` as the synchronization point. A client updates `attach`, optionally updates `damage`, then writes a new commit serial to `commit`.

Bloom reads the committed attachment, maps the memfd, copies the pixels into the compositor scene, and mirrors the result into `status/`.

### Configure Events

Bloom appends newline-delimited JSON objects to `events`.

Example:

```json
{"type":"configure","serial":42,"width":800,"height":600,"states":["activated","maximized"]}
```

Bloom updates `status/last_configure_serial` at the same time.

### Requested vs Current State

Client preference lives under `shell/requested/`.

Compositor-owned state lives under `shell/current/` and `status/`.

Clients request fullscreen or maximize by writing `1` to the requested files. Bloom observes those changes, updates current state, and emits a configure event.

### Binding

`bind/surface` contains the surface id bound to the window. A window may exist without a bound surface. Bloom keeps such windows unmapped until a committed surface arrives.

### Destroy Semantics

Directory removal is client-initiated destroy in the current implementation. When `/session/windows/<wid>` or `/session/surfaces/<sid>` disappears, Bloom tears down the corresponding internal object.

Compositor-initiated close remains an event-stream action: Bloom should append `{"type":"close"}` to `events` when it requests that a client close.

## Watch Semantics

Bloom uses kernel-native VFS watches for object lifecycle and durable state changes.

Global watches:

- `/session/windows/`
- `/session/surfaces/`

Per-window watches:

- `/session/windows/<wid>/shell/`
- `/session/windows/<wid>/shell/requested/`
- `/session/windows/<wid>/bind/`

Per-surface watches:

- `/session/surfaces/<sid>/`

These watches wake Bloom for:

- window creation and removal
- surface creation and removal
- title and app_id changes
- requested fullscreen/maximize changes
- bind and rebind changes
- attach/commit/damage file updates

Watches are not meant to replace live protocol traffic. They are used for object lifecycle and durable state updates. `events`, `attach`, and `commit` remain the conversational protocol files.

## Security Model

Access to `/session/windows` and `/session/surfaces` is the capability to request visible UI state.

The intended rules are:

- only authorized processes may create or mutate session GUI objects
- clients may not overwrite compositor-owned `current/*` or `status/*`
- surface binding must be permission-checked so one client cannot steal another client’s surface
- destroy and rebind operations must be authorized

The current implementation establishes the object model and watch flow first. Stronger permission checks still need kernel and session policy work.

## Examples

### Create a Window Object

```text
mkdir /session/windows/100
mkdir /session/windows/100/shell
mkdir /session/windows/100/shell/requested
mkdir /session/windows/100/shell/current
mkdir /session/windows/100/bind
mkdir /session/windows/100/status
echo "demo.app" > /session/windows/100/shell/app_id
echo "Demo" > /session/windows/100/shell/title
```

Bloom discovers the new object through the `/session/windows/` watch and begins managing it.

### Bind a Surface

```text
mkdir /session/surfaces/200
mkdir /session/surfaces/200/status
printf "fd=57\nwidth=320\nheight=180\nstride=1280\nformat=1\n" > /session/surfaces/200/attach
echo "1" > /session/surfaces/200/commit
echo "200" > /session/windows/100/bind/surface
```

Bloom notices the bind and the committed surface through VFS watches, maps the memfd, and presents the surface.

### Read Configure Events

Read the appended JSON lines from `/session/windows/<wid>/events` and track the latest `status/last_configure_serial`.

### Request Fullscreen

```text
echo "1" > /session/windows/100/shell/requested/fullscreen
```

Bloom observes the requested-state change via a watch, updates `shell/current/fullscreen`, and appends a configure event describing the new state.
