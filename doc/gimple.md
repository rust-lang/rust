# GIMPLE

You can see the full documentation about what GIMPLE is [here](https://gcc.gnu.org/onlinedocs/gccint/GIMPLE.html). In this document we will explain how to generate it.

First, we'll copy the content from `gcc/gcc/testsuite/jit.dg/test-const-attribute.c` into a
file named `local.c` and remove the content we're not interested into:

```diff
- /* { dg-do compile { target x86_64-*-* } } */
...
- /* We don't want set_options() in harness.h to set -O3 to see that the const
-    attribute affects the optimizations. */
- #define TEST_ESCHEWS_SET_OPTIONS
- static void set_options (gcc_jit_context *ctxt, const char *argv0)
- {
-   // Set "-O3".
-   gcc_jit_context_set_int_option(ctxt, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
- }
-
- #define TEST_COMPILING_TO_FILE
- #define OUTPUT_KIND      GCC_JIT_OUTPUT_KIND_ASSEMBLER
- #define OUTPUT_FILENAME  "output-of-test-const-attribute.c.s"
- #include "harness.h"
...
- /* { dg-final { jit-verify-output-file-was-created "" } } */
- /* Check that the loop was optimized away */
- /* { dg-final { jit-verify-assembler-output-not "jne" } } */
```

Then we'll add a `main` function which will call the `create_code` function but
also add the calls we need to generate the GIMPLE:

```C
int main() {
    gcc_jit_context *ctxt = gcc_jit_context_acquire();
    // To set `-O3`, update it depending on your needs.
    gcc_jit_context_set_int_option(ctxt, GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 3);
    // Very important option to generate the gimple format.
    gcc_jit_context_set_bool_option(ctxt, GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, 1);
    create_code(ctxt, NULL);

    gcc_jit_context_compile(ctxt);
    // If you want to compile to assembly (or any other format) directly, you can
    // use the following call instead:
    // gcc_jit_context_compile_to_file(ctxt, GCC_JIT_OUTPUT_KIND_ASSEMBLER, "out.s");

    return 0;
}
```

Then we can compile it by using:

```console
gcc local.c -I `pwd`/gcc/gcc/jit/ -L `pwd`/gcc-build/gcc -lgccjit -o out
```

And finally when you run it:

```console
LD_LIBRARY_PATH=`pwd`/gcc-build/gcc LIBRARY_PATH=`pwd`/gcc-build/gcc ./out
```

It should display:

```c
__attribute__((const))
int xxx ()
{
  int D.3394;
  int sum;
  int x;

  <D.3377>:
  x = 45;
  sum = 0;
  goto loop_cond;
  loop_cond:
  x = x >> 1;
  if (x != 0) goto after_loop; else goto loop_body;
  loop_body:
  _1 = foo (x);
  _2 = _1 * 2;
  x = x + _2;
  goto loop_cond;
  after_loop:
  D.3394 = sum;
  return D.3394;
}
```

An alternative way to generate the GIMPLE is to replace:

```c
    gcc_jit_context_set_bool_option(ctxt, GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, 1);
```

with:

```c
    gcc_jit_context_add_command_line_option(ctxt, "-fdump-tree-gimple");
```

(although you can have both at the same time too). Then you can compile it like previously. Only one difference: before executing it, I recommend to run:

```console
rm -rf /tmp/libgccjit-*
```

to make it easier for you to know which folder to look into.

Once the execution is done, you should now have a file with path looking like `/tmp/libgccjit-9OFqkD/fake.c.006t.gimple` which contains the GIMPLE format.
