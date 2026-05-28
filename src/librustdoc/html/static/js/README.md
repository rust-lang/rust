# Rustdoc JS

These JavaScript files are incorporated into the rustdoc binary at build time,
and are minified and written to the filesystem as part of the doc build process.

We use the [TypeScript Compiler](https://www.typescriptlang.org/docs/handbook/jsdoc-supported-types.html)
dialect of JSDoc to comment our code and annotate params and return types.
To run a check:

    npm i -g typescript
    tsc --project tsconfig.json
