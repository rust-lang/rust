// A regression test for #74713.

// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn bug<'a>()
where
    [(); { //[full]~ ERROR: mismatched types
        let _: &'a ();
        //[min]~^ ERROR: a non-static lifetime is not allowed in a `const`
    }]: ,
{
}

fn main() {}
