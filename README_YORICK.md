# Yorick-specific Notes

## LLVM Version

ykrustc requires a version of LLVM supporting code-gen for DILabels.

At the time of writing, this has not been merged into master. More details can
be found here:
https://reviews.llvm.org/D45045

Until this is merged, use the `master` branch in:
https://github.com/Hsiangkai/llvm.git

ykrustc was developed using version 4d9f26638007efe1c0dd8ccd689bad808df5a772.

Don't forget to update `llvm-config` in your config.toml.
