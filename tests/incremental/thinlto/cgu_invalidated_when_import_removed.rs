//@ revisions: cfail1 cfail2
//@ compile-flags: -O -Zhuman-readable-cgu-names -Cllvm-args=-import-instr-limit=10
//@ build-pass

// rust-lang/rust#59535:
//
// Consider a call-graph like `[A] -> [B -> D] <- [C]` (where the letters are
// functions and the modules are enclosed in `[]`)
//
// In our specific instance, the earlier compilations were inlining the call
// to`B` into `A`; thus `A` ended up with an external reference to the symbol `D`
// in its object code, to be resolved at subsequent link time. The LTO import
// information provided by LLVM for those runs reflected that information: it
// explicitly says during those runs, `B` definition and `D` declaration were
// imported into `[A]`.
//
// The change between incremental builds was that the call `D <- C` was removed.
//
// That change, coupled with other decisions within `rustc`, made the compiler
// decide to make `D` an internal symbol (since it was no longer accessed from
// other codegen units, this makes sense locally). And then the definition of
// `D` was inlined into `B` and `D` itself was eliminated entirely.
//
// The current LTO import information reported that `B` alone is imported into
// `[A]` for the *current compilation*. So when the Rust compiler surveyed the
// dependence graph, it determined that nothing `[A]` imports changed since the
// last build (and `[A]` itself has not changed either), so it chooses to reuse
// the object code generated during the previous compilation.
//
// But that previous object code has an unresolved reference to `D`, and that
// causes a link time failure!

fn main() {
    foo::foo();
    bar::baz();
}

mod foo {

    // In cfail1, foo() gets inlined into main.
    // In cfail2, ThinLTO decides that foo() does not get inlined into main, and
    // instead bar() gets inlined into foo(). But faulty logic in our incr.
    // ThinLTO implementation thought that `main()` is unchanged and thus reused
    // the object file still containing a call to the now non-existent bar().
    pub fn foo(){
        bar()
    }

    // This function needs to be big so that it does not get inlined by ThinLTO
    // but *does* get inlined into foo() once it is declared `internal` in
    // cfail2.
    pub fn bar(){
        println!("quux1");
        println!("quux2");
        println!("quux3");
        println!("quux4");
        println!("quux5");
        println!("quux6");
        println!("quux7");
        println!("quux8");
        println!("quux9");
    }
}

mod bar {

    #[inline(never)]
    pub fn baz() {
        #[cfg(cfail1)]
        {
            crate::foo::bar();
        }
    }
}
