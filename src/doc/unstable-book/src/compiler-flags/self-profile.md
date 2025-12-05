# `self-profile`

--------------------

The `-Zself-profile` compiler flag enables rustc's internal profiler.
When enabled, the compiler will output three binary files in the specified directory (or the current working directory if no directory is specified).
These files can be analyzed by using the tools in the [`measureme`] repository.

To control the data recorded in the trace files, use the `-Zself-profile-events` flag.

For example:

First, run a compilation session and provide the `-Zself-profile` flag:

```console
$ rustc --crate-name foo -Zself-profile
```

This will generate three files in the working directory such as:

- `foo-1234.events`
- `foo-1234.string_data`
- `foo-1234.string_index`

Where `foo` is the name of the crate and `1234` is the process id of the rustc process.

To get a summary of where the compiler is spending its time:

```console
$ ../measureme/target/release/summarize summarize foo-1234
```

To generate a flamegraph of the same data:

```console
$ ../measureme/target/release/inferno foo-1234
```

To dump the event data in a Chromium-profiler compatible format:

```console
$ ../measureme/target/release/crox foo-1234
```

For more information, consult the [`measureme`] documentation.

[`measureme`]: https://github.com/rust-lang/measureme.git
