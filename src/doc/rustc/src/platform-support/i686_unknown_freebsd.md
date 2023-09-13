# `*-unknown-freebsd`

[FreeBSD] is an operating system used to power modern servers,
desktops, and embedded platforms.

| Target                        | Tier                 |
| ----------------------------- | -------------------- |
| `aarch64-unknown-freebsd`     | Tier 3               |
| `armv6-unknown-freebsd`       | Tier 3               |
| `armv7-unknown-freebsd`       | Tier 3               |
| `x86_64-unknown-freebsd`      | Tier 2 w/ Host Tools |
| `i686-unknown-freebsd`        | Tier 2               |
| `powerpc64-unknown-freebsd`   | Tier 3               |
| `powerpc64le-unknown-freebsd` | Tier 3               |
| `powerpc-unknown-freebsd`     | Tier 3               |
| `riscv64gc-unknown-freebsd`   | Tier 3               |

## Target maintainers

- rust@FreeBSD.org

## Requirements

Missing!

## Building the target

The target can be built by enabling it for a `rustc` build.

```toml
[build]
target = ["$ARCH-unknown-freebsd"]

[target.$ARCH-unknown-openbsd]
cc = "$ARCH-freebsd-cc"
```

## Building Rust programs

Missing!

## Testing

Missing!

## Cross-compilation toolchains and C code

Missing!

[FreeBSD]: https://www.freebsd.org
