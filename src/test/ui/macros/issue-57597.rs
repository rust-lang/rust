// Regression test for #57597.
//
// Make sure that nested matchers work correctly rather than causing an infinite loop or crash.

// edition:2018

macro_rules! foo1 {
    ($($($i:ident)?)+) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo2 {
    ($($($i:ident)?)*) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo3 {
    ($($($i:ident)?)?) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo4 {
    ($($($($i:ident)?)?)?) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo5 {
    ($($($($i:ident)*)?)?) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo6 {
    ($($($($i:ident)?)*)?) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo7 {
    ($($($($i:ident)?)?)*) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo8 {
    ($($($($i:ident)*)*)?) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo9 {
    ($($($($i:ident)?)*)*) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo10 {
    ($($($($i:ident)?)*)+) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo11 {
    ($($($($i:ident)+)?)*) => {};
    //~^ ERROR repetition matches empty token tree
}

macro_rules! foo12 {
    ($($($($i:ident)+)*)?) => {};
    //~^ ERROR repetition matches empty token tree
}

fn main() {
    foo1!();
    foo2!();
    foo3!();
    foo4!();
    foo5!();
    foo6!();
    foo7!();
    foo8!();
    foo9!();
    foo10!();
    foo11!();
    foo12!();
}
