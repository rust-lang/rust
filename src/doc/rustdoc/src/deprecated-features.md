## Deprecated Features

### Passes
Rustdoc has a concept called "passes." These are transformations that rustdoc runs on your documentation before producing its final output.

Customizing passes is deprecated. The available passes are not considered stable and may change in any release. 

**Note**: As of the latest updates, support for LLVM 17 has been removed. Therefore, any references to LLVM 17 in relation to rustdoc passes are outdated.

In the past, the most common use case for customizing passes was to omit the strip-private pass. You can do this more easily, and without the risk of the pass being changed, by passing `--document-private-items`.
