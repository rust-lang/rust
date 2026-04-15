# ThingOS Ontology

ThingOS is a **typed-world operating system**.

Its canonical architectural truth is not Unix process/filesystem vocabulary,
but a set of first-class concepts that describe what the system **is** before
any compatibility surface is projected over it.

Unix compatibility remains important, but it is a **projection layer** over the
canonical model, not the ontology itself.

---

## Canonical concepts

### Thing

A **Thing** is any first-class object in the system that can be:

- referred to
- related to other objects
- acted upon
- observed
- projected into one or more forms

A Thing is the basic referent of the system.

Examples, depending on maturity of the implementation, may include:

- a file-like object
- a device
- a channel endpoint
- a process-like lifecycle object
- a display surface
- a directory-like container
- a future graph object or world object

A Thing is **not** defined by one particular interface such as a path or a file
descriptor. Those are merely access forms.

### Kind

A **Kind** is the authoritative definition of structure and meaning.

Kinds answer:

- what something is
- what shape its data has
- what operations are meaningful on it
- what compatibility promises exist over time

A Thing has a Kind.

Kinds are architectural truth. They are not ad hoc wire formats or accidental
Rust structs.

### Form

A **Form** is a representation or projection of a Kind.

Kinds answer **what** something is.
Forms answer **how** it is expressed.

A single Thing/Kind may have multiple Forms, such as:

- binary IPC/storage form
- filesystem-visible form
- debug text form
- JSON bridge form
- ioctl/message payload form

Form exists to prevent the system from collapsing into the assumption that one
surface representation is the object itself.

### Place

A **Place** is a context of visibility, relation, and encounter.

A Place determines:

- which Things are visible from a context
- how they are related or named there
- what world or namespace is being inhabited
- what projections are available

Filesystem namespace, cwd, root, session world, and future graph/world
projections are all best understood as aspects or projections of Place.

A Place is not merely a path string.

### Person

A **Person** is an actor in the system.

A Person is the entity on whose behalf action is taken. Person is the canonical
subject that:

- acts
- inhabits
- perceives
- is granted or denied power

Unix credentials may eventually project some of this, but they are not the full
meaning of Person.

### Authority

**Authority** is the context of legitimate action.

Authority answers:

- what a Person may do
- through what capabilities or permissions action is allowed
- which references, operations, or projections are permitted

Authority is distinct from Person:

- Person answers **who acts**
- Authority answers **with what power they act**

Unix uid/gid/permission semantics are compatibility projections in this area,
not the final truth.

### Presence

**Presence** is the embodied participation of a Person in a Place.

Presence answers:

- how a Person is attached to a world or interface
- what terminal/session/window/embodiment semantics apply
- how foreground/background or controlling-interface style semantics should be understood

Presence is distinct from Place:

- Place answers **where**
- Presence answers **how one is present there**

Presence is also distinct from Authority:

- Authority answers **what may be done**
- Presence answers **how the actor is situated**

This is the likely canonical home for semantics that Unix expresses through
session, controlling tty, and related attachment concepts.

---

## Relationship summary

These are the core relations:

- A **Thing** has a **Kind**.
- A **Kind** may be expressed in multiple **Forms**.
- A **Thing** may be encountered in one or more **Places**.
- A **Person** acts through **Authority**.
- A **Person** inhabits a **Place** via **Presence**.
- A **Place** may project Things into filesystem, message, debug, or other Forms.

Put differently:

- **Thing** is the referent.
- **Kind** is the meaning.
- **Form** is the expression.
- **Place** is the world-context.
- **Person** is the actor.
- **Authority** is the power.
- **Presence** is the embodiment.

---

## Canonical vs compatibility concepts

The following concepts are canonical architectural truth:

- Thing
- Kind
- Form
- Place
- Person
- Authority
- Presence

The following concepts are important, but are generally **projection or
compatibility-facing** and must not silently redefine ontology:

- path
- file descriptor
- process
- signal
- session
- process group
- cwd
- mount namespace

The following concepts are currently useful **transitional decomposition seams**:

- Job
- Space
- Task / Thread scheduling identities (subject to exact long-term naming)

---

## Projection rule

**New kernel meaning must be introduced in canonical typed-world terms first.**

Only after the canonical object is understood should a Unix-visible projection
be designed.

Examples:

- Do not introduce new architectural truth directly as a `process` feature if
  it is actually about lifecycle, authority, or place.
- Do not treat a path as the thing itself; it is one way a Thing is encountered
  in a Place.
- Do not treat a file descriptor as ontology; it is one reference form.
- Do not let signals become the general event model; they are a compatibility surface.

---

## Concrete reinterpretations of current systems

### VFS

The VFS is a strong and important subsystem, but it is not the ontology of the
system.

It should be understood as:

- one projection mechanism for Things
- one access form for Things
- one projection of Place

A path is therefore not the essence of an object. It is one Place-relative way
of reaching a Thing.

### Process

`Process` is not the final metaphysical center of ThingOS.

The current code already decomposes process-shaped responsibilities into:

- lifecycle-like concerns
- address-space concerns
- Unix compatibility concerns
- place-like concerns

This should be interpreted as evidence that `process` is a compatibility-facing
projection rather than the true object model.

### File descriptor / handle

An fd is one form of Thing reference.

It is useful and important, but should not be treated as the identity of the
Thing itself.

### Signal

A signal is a Unix projection of notification/event semantics.

Canonical event semantics should ultimately be expressed in typed messages,
relations, and explicit lifecycle/presence coordination rather than relying on
Unix signal semantics as truth.

---

## Review rule

When reviewing or designing a change, always ask:

1. What is the canonical concept here?
2. Is this change introducing truth or compatibility?
3. If compatibility, what canonical concept is it projecting from?
4. Is Unix vocabulary being used because it is required, or because it is familiar?

If the answer begins and ends in Unix terminology, the design is probably still
standing in the projection layer rather than the ontology.
