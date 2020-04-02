# Profiling the compiler

This section talks about how to profile the compiler and find out where it spends its time.  

Depending on what you're trying to measure, there are several different approaches:

- If you want to see if a PR improves or regresses compiler performance:
  - The [rustc-perf](https://github.com/rust-lang/rustc-perf) project makes this easy and can be triggered to run on a PR via the `@rustc-perf` bot.
  
- If you want a medium-to-high level overview of where `rustc` is spending its time:
  - The `-Zself-profile` flag and [measureme](https://github.com/rust-lang/measureme) tools offer a query-based approach to profiling.
    See [their docs](https://github.com/rust-lang/measureme/blob/master/summarize/Readme.md) for more information.
    
- If you want function level performance data or even just more details than the above approaches:
  - Consider using a native code profiler such as [perf](profiling/with_perf.html).

