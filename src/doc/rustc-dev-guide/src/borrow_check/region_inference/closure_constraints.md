# Propagating closure constraints

When we are checking the type tests and universal regions, we may come
across a constraint that we can't prove yet if we are in a closure
body! However, the necessary constraints may actually hold (we just
don't know it yet). Thus, if we are inside a closure, we just collect
all the constraints we can't prove yet and return them. Later, when we
are borrow check the MIR node that created the closure, we can also
check that these constraints hold. At that time, if we can't prove
they hold, we report an error.
