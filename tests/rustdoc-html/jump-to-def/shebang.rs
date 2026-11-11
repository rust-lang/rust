#!/path/to/my/interpreter
//@ compile-flags: -Zunstable-options --generate-link-to-definition

// Ensure that we can successfully generate links to definitions in the presence of shebang.
// Implementation-wise, shebang is not a token that's emitted by the lexer. Instead, we need
// to offset the actual lexing which is tricky due to all the byte index and span calculations
// in the Classifier.

fn scope() {
//@ has 'src/shebang/shebang.rs.html'
//@ has - '//a[@href="#15"]' 'function'
    function();
}

fn function() {}
