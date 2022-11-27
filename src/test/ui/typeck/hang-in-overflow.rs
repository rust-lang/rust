// normalize-stderr-test "the requirement `.*`" -> "the requirement `...`"
// normalize-stderr-test "required for `.*` to implement `.*`" -> "required for `...` to implement `...`"
// normalize-stderr-test: ".*the full type name has been written to.*\n" -> ""

// Currently this fatally aborts instead of hanging.
// Make sure at least that this doesn't turn into a hang.

fn f() {
    foo::<_>();
    //~^ ERROR overflow evaluating the requirement
}

fn foo<B>()
where
    Vec<[[[B; 1]; 1]; 1]>: PartialEq<B>,
{
}

fn main() {}
