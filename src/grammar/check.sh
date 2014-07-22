#!/bin/sh

# Run the reference lexer against libsyntax and compare the tokens and spans.
# If "// ignore-lexer-test" is present in the file, it will be ignored.


# Argument $1 is the file to check, $2 is the classpath to use, $3 is the path
# to the grun binary, $4 is the path to the verify binary, $5 is the path to
# RustLexer.tokens
if [ "${VERBOSE}" == "1" ]; then
    set -x
fi

check() {
    grep --silent "// ignore-lexer-test" $1;

    # if it's *not* found...
    if [ $? -eq 1 ]; then
        cd $2 # This `cd` is so java will pick up RustLexer.class. I couldn't
        # figure out how to wrangle the CLASSPATH, just adding build/grammr didn't
        # seem to have anny effect.
        if $3 RustLexer tokens -tokens < $1 | $4 $1 $5; then
            echo "pass: $1"
        else
            echo "fail: $1"
        fi
    else
        echo "skip: $1"
    fi
}

for file in $(find $1 -iname '*.rs' ! -path '*/test/compile-fail*'); do
    check $file $2 $3 $4 $5
done
