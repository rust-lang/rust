# Reporting backend crashes

If after a compilation failure you are greeted by a large amount of llvm-ir code, then our enzyme backend likely failed to compile your code. These cases are harder to debug, so your help is highly appreciated. Please also keep in mind that release builds are usually much more likely to work at the moment.

The final goal here is to reproduce your bug in the enzyme [compiler explorer](https://enzyme.mit.edu/explorer/), in order to create a bug report in the [Enzyme](https://github.com/enzymead/enzyme/issues) repository.

We have an `autodiff` flag which you can pass to `rustflags` to help with this. it will print the whole llvm-ir module, along with some `__enzyme_fwddiff` or `__enzyme_autodiff` calls. A potential workflow on linux could look like:  

## Controlling llvm-ir generation

Before generating the llvm-ir, keep in mind two techniques that can help ensure the relevant rust code is visible for debugging:

- **`std::hint::black_box`**: wrap rust variables or expressions in `std::hint::black_box()` to prevent rust and llvm from optimizing them away. This is useful when you need to inspect or manually manipulate specific values in the llvm-ir.
- **`extern "rust"` or `extern "c"`**: if you want to see how a specific function declaration is lowered to llvm-ir, you can declare it as `extern "rust"` or `extern "c"`. You can also look for existing `__enzyme_autodiff` or similar declarations within the generated module for examples.

## 1) Generate an llvm-ir reproducer

```sh
rustflags="-z autodiff=enable,printmodbefore" cargo +enzyme build --release &> out.ll 
```

This also captures a few warnings and info messages above and below your module. open out.ll and remove every line above `; moduleid = <somehash>`. Now look at the end of the file and remove everything that's not part of llvm-ir, i.e. remove errors and warnings. The last line of your llvm-ir should now start with `!<somenumber> = `, i.e. `!40831 = !{i32 0, i32 1037508, i32 1037538, i32 1037559}` or `!43760 = !dilocation(line: 297, column: 5, scope: !43746)`.

The actual numbers will depend on your code.  

## 2) Check your llvm-ir reproducer

To confirm that your previous step worked, we will use llvm's `opt` tool. find your path to the opt binary, with a path similar to `<some_dir>/rust/build/<x86/arm/...-target-tripple>/build/bin/opt`. also find `llvmenzyme-19.<so/dll/dylib>` path, similar to `/rust/build/target-tripple/enzyme/build/enzyme/llvmenzyme-19`. Please keep in mind that llvm frequently updates it's llvm backend, so the version number might be higher (20, 21, ...). Once you have both, run the following command:

```sh
<path/to/opt> out.ll -load-pass-plugin=/path/to/llvmenzyme-19.so -passes="enzyme" -s
```

If the previous step succeeded, you are going to see the same error that you saw when compiling your rust code with cargo. 

If you fail to get the same error, please open an issue in the rust repository. If you succeed, congrats! the file is still huge, so let's automatically minimize it.

## 3) Minimize your llvm-ir reproducer

First find your `llvm-extract` binary, it's in the same folder as your opt binary. then run:

```sh
<path/to/llvm-extract> -s --func=<name> --recursive --rfunc="enzyme_autodiff*" --rfunc="enzyme_fwddiff*" --rfunc=<fnc_called_by_enzyme> out.ll -o mwe.ll 
```

This command creates `mwe.ll`, a minimal working example.

Please adjust the name passed with the last `--func` flag. You can either apply the `#[no_mangle]` attribute to the function you differentiate, then you can replace it with the rust name. otherwise you will need to look up the mangled function name. To do that, open `out.ll` and search for `__enzyme_fwddiff` or `__enzyme_autodiff`. the first string in that function call is the name of your function. example:

```llvm-ir 
define double @enzyme_opt_helper_0(ptr %0, i64 %1, double %2) {
  %4 = call double (...) @__enzyme_fwddiff(ptr @_zn2ad3_f217h3b3b1800bd39fde3e, metadata !"enzyme_const", ptr %0, metadata !"enzyme_const", i64 %1, metadata !"enzyme_dup", double %2, double %2)
  ret double %4
}
```

Here, `_zn2ad3_f217h3b3b1800bd39fde3e` is the correct name. make sure to not copy the leading `@`. redo step 2) by running the `opt` command again, but this time passing `mwe.ll` as the input file instead of `out.ll`. Check if this minimized example still reproduces the crash.

## 4) (Optional) Minimize your llvm-ir reproducer further.

After the previous step you should have an `mwe.ll` file with ~5k loc. let's try to get it down to 50. find your `llvm-reduce` binary next to `opt` and `llvm-extract`. Copy the first line of your error message, an example could be:

```sh
opt: /home/manuel/prog/rust/src/llvm-project/llvm/lib/ir/instructions.cpp:686: void llvm::callinst::init(llvm::functiontype*, llvm::value*, llvm::arrayref<llvm::value*>, llvm::arrayref<llvm::operandbundledeft<llvm::value*> >, const llvm::twine&): assertion `(args.size() == fty->getnumparams() || (fty->isvararg() && args.size() > fty->getnumparams())) && "calling a function with bad signature!"' failed.
```

If you just get a `segfault` there is no sensible error message and not much to do automatically, so continue to 5).  
otherwise, create a `script.sh` file containing

```sh
#!/bin/bash
<path/to/your/opt> $1 -load-pass-plugin=/path/to/llvmenzyme-19.so -passes="enzyme" \
    |& grep "/some/path.cpp:686: void llvm::callinst::init"
```

Experiment a bit with which error message you pass to grep. it should be long enough to make sure that the error is unique. However, for longer errors including `(` or `)` you will need to escape them correctly which can become annoying. Run

```sh 
<path/to/llvm-reduce> --test=script.sh mwe.ll 
```

If you see `input isn't interesting! verify interesting-ness test`, you got the error message in script.sh wrong, you need to make sure that grep matches your actual error. If all works out, you will see a lot of iterations, ending with a new `reduced.ll` file. Verify with `opt` that you still get the same error.

### Advanced debugging: manual llvm-ir investigation

Once you have a minimized reproducer (`mwe.ll` or `reduced.ll`), you can delve deeper:

- **manual editing:** try manually rewriting the llvm-ir. for certain issues, like those involving indirect calls, you might investigate enzyme-specific intrinsics like `__enzyme_virtualreverse`. Understanding how to use these might require consulting enzyme's documentation or source code.
- **enzyme test cases:** look for relevant test cases within the [enzyme repository](https://github.com/enzymead/enzyme/tree/main/enzyme/test) that might demonstrate the correct usage of features or intrinsics related to your problem.

## 5) Report your bug.

Afterwards, you should be able to copy and paste your `mwe.ll` (or `reduced.ll`) example into our [compiler explorer](https://enzyme.mit.edu/explorer/).

- Select `llvm ir` as language and `opt 20` as compiler.
- Replace the field to the right of your compiler with `-passes="enzyme"`, if it is not already set.
- Hopefully, you will see once again your now familiar error.
- Please use the share button to copy links to them.
- Please create an issue on [https://github.com/enzymead/enzyme/issues](https://github.com/enzymead/enzyme/issues) and share `mwe.ll` and (if you have it) `reduced.ll`, as well as links to the compiler explorer. Please feel free to also add your rust code or a link to it.

#### Documenting findings

some enzyme errors, like `"attempting to call an indirect active function whose runtime value is inactive"`, have historically caused confusion. If you investigate such an issue, even if you don't find a complete solution, please consider documenting your findings. If the insights are general to enzyme and not specific to its rust usage, contributing them to the main [enzyme documentation](https://github.com/enzymead/www) is often the best first step. You can also mention your findings in the relevant enzyme github issue or propose updates to these docs if appropriate. This helps prevent others from starting from scratch.

With a clear reproducer and documentation, hopefully an enzyme developer will be able to fix your bug. Once that happens, the enzyme submodule inside the rust compiler will be updated, which should allow you to differentiate your rust code. Thanks for helping us to improve rust-ad.

# Minimize rust code

Beyond having a minimal llvm-ir reproducer, it is also helpful to have a minimal rust reproducer without dependencies. This allows us to add it as a test case to ci once we fix it, which avoids regressions for the future.

There are a few solutions to help you with minimizing the rust reproducer. This is probably the most simple automated approach: [cargo-minimize](https://github.com/nilstrieb/cargo-minimize).

Otherwise we have various alternatives, including [`treereduce`](https://github.com/langston-barrett/treereduce), [`halfempty`](https://github.com/googleprojectzero/halfempty), or [`picireny`](https://github.com/renatahodovan/picireny), potentially also [`creduce`](https://github.com/csmith-project/creduce).
