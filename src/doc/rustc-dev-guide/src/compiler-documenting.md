# Documenting rustc

You might want to build documentation of the various components 
available like the standard library. There’s two ways to go about this.
 You can run rustdoc directly on the file to make sure the HTML is 
 correct which is fast or you can build the documentation as part of the 
 build process through x.py. Both are viable methods since documentation 
 is more about the content.

## Document everything

   `./x.py doc`

## If you want to avoid the whole Stage 2 build

   `./x.py doc --stage 1`

First the compiler and rustdoc get built to make sure everything is okay 
and then it documents the files.

## Document specific components

```bash
   ./x.py doc src/doc/book
   ./x.py doc src/doc/nomicon
   ./x.py doc src/doc/book src/libstd
```

Much like individual tests or building certain components you can build only
 the documentation you want.

## Document internal rustc items
By default rustc does not build the compiler docs for its internal items. 
Mostly because this is useless for the average user. However, you might need 
to have it available so you can understand the types. Here’s how you can 
compile it yourself. From the top level directory where x.py is located run:

    cp config.toml.example config.toml

Next open up config.toml and make sure these two lines are set to true:

docs = true
compiler-docs = true
When you want to build the compiler docs as well run this command:

   `./x.py doc`

This will see that the docs and compiler-docs options are set to true 
and build the normally hidden compiler docs!