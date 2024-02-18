# Frequently asked questions

## Why is it called salsa?

I like salsa! Don't you?! Well, ok, there's a bit more to it. The
underlying algorithm for figuring out which bits of code need to be
re-executed after any given change is based on the algorithm used in
rustc. Michael Woerister and I first described the rustc algorithm in
terms of two colors, red and green, and hence we called it the
"red-green algorithm". This made me think of the New Mexico State
Question --- ["Red or green?"][nm] --- which refers to chile
(salsa). Although this version no longer uses colors (we borrowed
revision counters from Glimmer, instead), I still like the name.

[nm]: https://www.sos.state.nm.us/about-new-mexico/state-question/

## What is the relationship between salsa and an Entity-Component System (ECS)?

You may have noticed that Salsa "feels" a lot like an ECS in some
ways. That's true -- Salsa's queries are a bit like *components* (and
the keys to the queries are a bit like *entities*). But there is one
big difference: **ECS is -- at its heart -- a mutable system**. You
can get or set a component of some entity whenever you like. In
contrast, salsa's queries **define "derived values" via pure
computations**.

Partly as a consequence, ECS doesn't handle incremental updates for
you. When you update some component of some entity, you have to ensure
that other entities' components are updated appropriately.

Finally, ECS offers interesting metadata and "aspect-like" facilities,
such as iterating over all entities that share certain components.
Salsa has no analogue to that.

