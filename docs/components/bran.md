# Bran

**Bran** is the **Boot Runtime Abstraction** for ThingOS.

The name officially (and unofficially) stands for:

> **B**oot **R**esponsibility & **A**rchitecture **N**onsense
> or, if you must, **Boot Runtime Abstractio-N**

Bran exists to take all the *messy, architecture-specific, bootloader-specific nonsense* and package it into a **small, stable, boring interface** that the kernel proper can rely on.

If the kernel is *what* the system does,
Bran is *how the system wakes up*.

## Role

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

## Responsibilities

Bran is responsible for *everything that must happen before the kernel can think clearly*.

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

## Design Principles

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
