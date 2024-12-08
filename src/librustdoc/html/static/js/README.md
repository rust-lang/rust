# Rustdoc JS

These JavaScript files are incorporated into the rustdoc binary at build time,
and are minified and written to the filesystem as part of the doc build process.

We use the [Closure Compiler](https://github.com/google/closure-compiler/wiki/Annotating-JavaScript-for-the-Closure-Compiler)
dialect of JSDoc to comment our code and annotate params and return types.
To run a check:

    ./x.py doc library/std
    npm i -g google-closure-compiler
    google-closure-compiler -W VERBOSE \
      build/<YOUR PLATFORM>/doc/{search-index*.js,crates*.js} \
      src/librustdoc/html/static/js/{search.js,main.js,storage.js} \
      --externs src/librustdoc/html/static/js/externs.js >/dev/null
