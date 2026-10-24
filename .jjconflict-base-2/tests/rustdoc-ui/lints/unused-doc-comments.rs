// Ensure that lint `unused_doc_comments` also gets emitted by rustdoc, not just by rustc.
//@ check-pass

pub fn f</** */T>() { //~ WARN unused doc comment
    /** */ let (); //~ WARN unused doc comment
}

macro_rules! m { () => { pub fn g() {} } }

/** */ m!(); //~ WARN unused doc comment
