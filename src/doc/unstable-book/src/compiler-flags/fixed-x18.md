# `fixed-x18`

This option prevents the compiler from using the x18 register. It is only
supported on `aarch64`.

From the [ABI spec][arm-abi]:

> X18 is the platform register and is reserved for the use of platform ABIs.
> This is an additional temporary register on platforms that don't assign a
> special meaning to it.

This flag only has an effect when the x18 register would otherwise be considered
a temporary register. When the flag is applied, x18 is always a reserved
register.

This flag is intended for use with the shadow call stack sanitizer. Generally,
when that sanitizer is enabled, the x18 register is used to store a pointer to
the shadow stack. Enabling this flag prevents the compiler from overwriting the
shadow stack pointer with temporary data, which is necessary for the sanitizer
to work correctly.

Currently, the `-Zsanitizer=shadow-call-stack` flag is only supported on
platforms that always treat x18 as a reserved register, and the `-Zfixed-x18`
flag is not required to use the sanitizer on such platforms. However, the
sanitizer may be supported on targets where this is not the case in the future.
One way to do so now on Nightly compilers is to explicitly supply this `-Zfixed-x18`
flag with `aarch64` targets, so that the sanitizer is available for instrumentation
on targets like `aarch64-unknown-none`, for instance. However, discretion is still
required to make sure that the runtime support is in place for this sanitizer
to be effective.

It is undefined behavior for `-Zsanitizer=shadow-call-stack` code to call into
code where x18 is a temporary register. On the other hand, when you are *not*
using the shadow call stack sanitizer, compilation units compiled with and
without the `-Zfixed-x18` flag are compatible with each other.

[arm-abi]: https://developer.arm.com/documentation/den0024/a/The-ABI-for-ARM-64-bit-Architecture/Register-use-in-the-AArch64-Procedure-Call-Standard/Parameters-in-general-purpose-registers
