# ThingOS

ThingOS treats **types as the system itself**.

Every meaningful element - objects, messages, files, interfaces, events - is defined by a single canonical schema called a **Kind**. That definition is used everywhere: to generate code, to validate data at runtime, to describe APIs, and to ensure that what the system *means* is consistent across all layers.

There is no split between "types in code," "formats on the wire," and "data on disk."
There is only the **Kind**, expressed in different **Forms**.

---

# The Model

Everything in ThingOS is a **Thing**, and every Thing has a **Kind**.

Kinds define:

* structure
* meaning
* valid operations
* compatibility over time

From this, the rest of the system emerges.

---

# People, Places, Things

The system is organized around three primary concepts:

* **Thing** - any object in the system
* **Place** - a context in which Things exist and interact
* **Person** - an actor that can act upon Things

A Person operates concretely:

> A **Person** acts through an **Authority**, inhabits a **Place** via a **Presence**, and manipulates **Things**.

These are not metaphors. They are first-class, typed objects.

---

# Execution

Computation is expressed through simple, orthogonal Things:

* **Task** - something that runs
* **Job** - something that lives and dies
* **Group** - something that coordinates

Each has a single responsibility. No hidden coupling. No overloaded abstractions.

---

# Kinds

A **Kind** is the authoritative definition of structure and meaning.

Kinds are:

* written once
* compiled into Rust types and interfaces
* enforced by the kernel
* available at runtime for introspection

They define:

* object layouts
* message schemas
* syscall arguments and results
* event payloads
* service contracts

Everything that crosses a boundary declares its Kind.

---

# Forms

A **Form** is a representation of a Kind.

Kinds answer *what something is*.
Forms answer *how it is expressed*.

A Kind may have multiple Forms:

* a canonical binary form for IPC and storage
* a debug text form for inspection
* bridge forms (such as JSON) for interoperability

Meaning stays fixed. Representation can vary.

---

# Toolchain

ThingOS uses a forked Rust toolchain with a custom target. The standard library is rebuilt for the system, and core types are generated from Kind definitions.

Rust provides the implementation surface.
Kinds provide the definition of truth.

---

# Direction

The system moves toward:

* eliminating duplicated type systems across layers
* generating interfaces instead of hand-writing them
* enforcing structure at system boundaries
* making the running system introspectable in its own terms

The end state is a system where the definitions used to build it are the same definitions it uses to understand itself.

---

# Summary

ThingOS is a **typed world** where:

* every Thing has a Kind
* every boundary is validated
* every interface is derived
* and meaning is consistent from compiler to kernel to runtime

The system is not just implemented in a language.

It *is* a language.
