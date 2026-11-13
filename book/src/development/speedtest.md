# Speedtest
`SPEEDTEST` is the tool we use to measure lint's performance, it works by executing the same test several times.

It's useful for measuring changes to current lints and deciding if the performance changes too much. `SPEEDTEST` is
accessed by the `SPEEDTEST` (and `SPEEDTEST_*`) environment variables.

## Checking Speedtest

To do a simple speed test of a lint (e.g. `allow_attributes`), use this command.

```sh
$ SPEEDTEST=ui TESTNAME="allow_attributes" cargo uitest
```

This will test all `ui` tests (`SPEEDTEST=ui`) whose names start with `allow_attributes`. By default, `SPEEDTEST` will
iterate your test 1000 times. But you can change this with `SPEEDTEST_ITERATIONS`.

```sh
$ SPEEDTEST=toml SPEEDTEST_ITERATIONS=100 TESTNAME="semicolon_block" cargo uitest
```
