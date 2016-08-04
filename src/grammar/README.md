Reference grammar.

Uses [antlr4](http://www.antlr.org/) and a custom Rust tool to compare
ASTs/token streams generated. You can use the `check-lexer` make target to
run all of the available tests.

To use manually, assuming antlr4 ist installed at `/usr/share/java/antlr-complete.jar`:

```
antlr4 RustLexer.g4
javac -classpath /usr/share/java/antlr-complete.jar *.java
rustc -O verify.rs
for file in ../*/**.rs; do
    echo $file;
    grun RustLexer tokens -tokens < "$file" | ./verify "$file" RustLexer.tokens || break
done
```

Note That the `../*/**.rs` glob will match every `*.rs` file in the above
directory and all of its recursive children. This is a zsh extension.
