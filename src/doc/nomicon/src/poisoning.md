# Poisoning

Although all unsafe code *must* ensure it has minimal exception safety, not all
types ensure *maximal* exception safety. Even if the type does, your code may
ascribe additional meaning to it. For instance, an integer is certainly
exception-safe, but has no semantics on its own. It's possible that code that
panics could fail to correctly update the integer, producing an inconsistent
program state.

This is *usually* fine, because anything that witnesses an exception is about
to get destroyed. For instance, if you send a Vec to another thread and that
thread panics, it doesn't matter if the Vec is in a weird state. It will be
dropped and go away forever. However some types are especially good at smuggling
values across the panic boundary.

These types may choose to explicitly *poison* themselves if they witness a panic.
Poisoning doesn't entail anything in particular. Generally it just means
preventing normal usage from proceeding. The most notable example of this is the
standard library's Mutex type. A Mutex will poison itself if one of its
MutexGuards (the thing it returns when a lock is obtained) is dropped during a
panic. Any future attempts to lock the Mutex will return an `Err` or panic.

Mutex poisons not for true safety in the sense that Rust normally cares about. It
poisons as a safety-guard against blindly using the data that comes out of a Mutex
that has witnessed a panic while locked. The data in such a Mutex was likely in the
middle of being modified, and as such may be in an inconsistent or incomplete state.
It is important to note that one cannot violate memory safety with such a type
if it is correctly written. After all, it must be minimally exception-safe!

However if the Mutex contained, say, a BinaryHeap that does not actually have the
heap property, it's unlikely that any code that uses it will do
what the author intended. As such, the program should not proceed normally.
Still, if you're double-plus-sure that you can do *something* with the value,
the Mutex exposes a method to get the lock anyway. It *is* safe, after all.
Just maybe nonsense.
