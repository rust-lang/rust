// These are supported by rustc syntactically but not semantically.

#[cfg(any())]
unsafe mod m { }

#[cfg(any())]
unsafe extern "C++" { }
