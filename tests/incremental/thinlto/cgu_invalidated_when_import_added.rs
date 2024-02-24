//@ revisions: cfail1 cfail2
//@ compile-flags: -O -Zhuman-readable-cgu-names -Cllvm-args=-import-instr-limit=10
//@ build-pass

// rust-lang/rust#59535:
//
// This is analogous to cgu_invalidated_when_import_removed.rs, but it covers
// the other direction:
//
// We start with a call-graph like `[A] -> [B -> D] [C]` (where the letters are
// functions and the modules are enclosed in `[]`), and add a new call `D <- C`,
// yielding the new call-graph: `[A] -> [B -> D] <- [C]`
//
// The effect of this is that the compiler previously classfied `D` as internal
// and the import-set of `[A]` to be just `B`. But after adding the `D <- C` call,
// `D` is no longer classified as internal, and the import-set of `[A]` becomes
// both `B` and `D`.
//
// We check this case because an early proposed pull request included an
// assertion that the import-sets monotonically decreased over time, a claim
// which this test case proves to be false.

fn main() {
    foo::foo();
    bar::baz();
}

mod foo {

    // In cfail1, ThinLTO decides that foo() does not get inlined into main, and
    // instead bar() gets inlined into foo().
    // In cfail2, foo() gets inlined into main.
    pub fn foo(){
        bar()
    }

    // This function needs to be big so that it does not get inlined by ThinLTO
    // but *does* get inlined into foo() when it is declared `internal` in
    // cfail1 (alone).
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
        #[cfg(cfail2)]
        {
            crate::foo::bar();
        }
    }
}
