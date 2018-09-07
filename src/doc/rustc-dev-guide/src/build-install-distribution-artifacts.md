# Build distribution artifacts

You might want to build and package up the compiler for distribution.
You’ll want to run this command to do it:

   `./x.py dist`

Other Flags

The same flags from build are available here. 
You might want to consider adding on the -j flag for faster builds 
when building a distribution artifact.

```
-j, --jobs JOBS     number of jobs to run in parallel
```


# Install distribution artifacts

If you’ve built a distribution artifact you might want to install it and 
test that it works on your target system. You’ll want to run this command:

   `./x.py install`

Other Flags
The same flags from build are available 