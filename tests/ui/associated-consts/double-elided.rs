struct S;

impl S {
    const C: &&str = &"";
    //~^ WARN `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| WARN `&` without an explicit lifetime name cannot be used here
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| ERROR in type `&&str`, reference has a longer lifetime than the data it references
}

fn main() {}
