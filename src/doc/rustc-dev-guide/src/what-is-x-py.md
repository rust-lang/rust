# what is x.py?

x.py is the script used to orchestrate the tooling in the rustc repository. 
It is the script that can build docs, run tests, and compile rustc. 
It is the now preferred way to build rustc and it replaces the old makefiles 
from before. Below are the different ways to utilize x.py in order to 
effectively deal with the repo for various common tasks.

### Build Flags

There are other flags you can pass to the build portion of x.py that can be 
beneficial to cutting down compile times or fitting other things you might 
need to change. They are:

```
Options:
    -v, --verbose       use verbose output (-vv for very verbose)
    -i, --incremental   use incremental compilation
        --config FILE   TOML configuration file for build
        --build BUILD   build target of the stage0 compiler
        --host HOST     host targets to build
        --target TARGET target targets to build
        --on-fail CMD   command to run on failure
        --stage N       stage to build
        --keep-stage N  stage to keep without recompiling
        --src DIR       path to the root of the rust checkout
    -j, --jobs JOBS     number of jobs to run in parallel
    -h, --help          print this help message
```

Note that the options --incremental, --keep-stage 0 and --jobs JOBS can be 
used in tandem with --stage to help reduce build times significantly by 
reusing already built components, reusing the first bootstrapped stage, and 
running compilation in parallel. To test changes you could run something like:

```bash
   ./x.py build --stage 1 --keep-stage 0 -j 4 -i
```

Please follow the links to build, document, test, benchmark and install 
distribution
 artifacts for rustc respectively.
