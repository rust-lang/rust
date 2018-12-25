#![allow(unused_macros)]

macro_rules! foo {
    ( $()* ) => {};
    //~^ ERROR repetition matches empty token tree
    ( $()+ ) => {};
    //~^ ERROR repetition matches empty token tree

    ( $(),* ) => {}; // PASS
    ( $(),+ ) => {}; // PASS

    ( [$()*] ) => {};
    //~^ ERROR repetition matches empty token tree
    ( [$()+] ) => {};
    //~^ ERROR repetition matches empty token tree

    ( [$(),*] ) => {}; // PASS
    ( [$(),+] ) => {}; // PASS

    ( $($()* $(),* $(a)* $(a),* )* ) => {};
    //~^ ERROR repetition matches empty token tree
    ( $($()* $(),* $(a)* $(a),* )+ ) => {};
    //~^ ERROR repetition matches empty token tree

    ( $(a     $(),* $(a)* $(a),* )* ) => {}; // PASS
    ( $($(a)+ $(),* $(a)* $(a),* )+ ) => {}; // PASS

    ( $(a $()+)* ) => {};
    //~^ ERROR repetition matches empty token tree
    ( $(a $()*)+ ) => {};
    //~^ ERROR repetition matches empty token tree
}


// --- Original Issue --- //

macro_rules! make_vec {
    (a $e1:expr $($(, a $e2:expr)*)*) => ([$e1 $($(, $e2)*)*]);
    //~^ ERROR repetition matches empty token tree
}

fn main() {
    let _ = make_vec![a 1, a 2, a 3];
}


// --- Minified Issue --- //

macro_rules! m {
    ( $()* ) => {}
    //~^ ERROR repetition matches empty token tree
}

m!();
