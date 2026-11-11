# `self-profile-events`

---------------------

The `-Zself-profile-events` compiler flag controls what events are recorded by the self-profiler when it is enabled via the `-Zself-profile` flag.

This flag takes a comma delimited list of event types to record.

For example:

```console
$ rustc -Zself-profile -Zself-profile-events=default,args
```

## Event types

- `query-provider`
  - Traces each query used internally by the compiler.

- `generic-activity`
  - Traces other parts of the compiler not covered by the query system.

- `query-cache-hit`
  - Adds tracing information that records when the in-memory query cache is "hit" and does not need to re-execute a query which has been cached.
  - Disabled by default because this significantly increases the trace file size.

- `query-blocked`
  - Tracks time that a query tries to run but is blocked waiting on another thread executing the same query to finish executing.
  - Query blocking only occurs when the compiler is built with parallel mode support.

- `incr-cache-load`
  - Tracks time that is spent loading and deserializing query results from the incremental compilation on-disk cache.

- `query-keys`
  - Adds a serialized representation of each query's query key to the tracing data.
  - Disabled by default because this significantly increases the trace file size.

- `function-args`
  - Adds additional tracing data to some `generic-activity` events.
  - Disabled by default for parity with `query-keys`.

- `llvm`
  - Adds tracing information about LLVM passes and codegeneration.
  - Disabled by default because this significantly increases the trace file size.

## Event synonyms

- `none`
  - Disables all events.
  Equivalent to the self-profiler being disabled.

- `default`
  - The default set of events which stikes a balance between providing detailed tracing data and adding additional overhead to the compilation.

- `args`
  - Equivalent to `query-keys` and `function-args`.

- `all`
  - Enables all events.

## Examples

Enable the profiler and capture the default set of events (both invocations are equivalent):

```console
$ rustc -Zself-profile
$ rustc -Zself-profile -Zself-profile-events=default
```

Enable the profiler and capture the default events and their arguments:

```console
$ rustc -Zself-profile -Zself-profile-events=default,args
```
