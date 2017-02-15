# The Perils Of Ownership Based Resource Management (OBRM)

OBRM (AKA RAII: Resource Acquisition Is Initialization) is something you'll
interact with a lot in Rust. Especially if you use the standard library.

Roughly speaking the pattern is as follows: to acquire a resource, you create an
object that manages it. To release the resource, you simply destroy the object,
and it cleans up the resource for you. The most common "resource" this pattern
manages is simply *memory*. `Box`, `Rc`, and basically everything in
`std::collections` is a convenience to enable correctly managing memory. This is
particularly important in Rust because we have no pervasive GC to rely on for
memory management. Which is the point, really: Rust is about control. However we
are not limited to just memory. Pretty much every other system resource like a
thread, file, or socket is exposed through this kind of API.
