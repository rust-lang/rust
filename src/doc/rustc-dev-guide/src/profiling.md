# Profiling the compiler

This discussion talks about how profile the compiler and find out
where it spends its time.  If you just want to get a general overview,
it is often a good idea to just add `-Zself-profile` option to the
rustc command line. This will break down time spent into various
categories.  But if you want a more detailed look, you probably want
to break out a custom profiler.

