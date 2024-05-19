The requested ABI is unsupported by the current target.

The rust compiler maintains for each target a list of unsupported ABIs on
that target. If an ABI is present in such a list this usually means that the
target / ABI combination is currently unsupported by llvm.

If necessary, you can circumvent this check using custom target specifications.
