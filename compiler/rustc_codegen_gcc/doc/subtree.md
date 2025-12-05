# git subtree sync

`rustc_codegen_gcc` is a subtree of the rust compiler. As such, it needs to be
sync from time to time to ensure changes that happened on their side are also
included on our side.

### How to install a forked git-subtree

Using git-subtree with `rustc` requires a patched git to make it work.
The PR that is needed is [here](https://github.com/gitgitgadget/git/pull/493).
Use the following instructions to install it:

```bash
git clone git@github.com:tqc/git.git
cd git
git checkout tqc/subtree
make
make install
cd contrib/subtree
make
cp git-subtree ~/bin
```

### Syncing with rust compiler

Do a sync with this command:

```bash
PATH="$HOME/bin:$PATH" ~/bin/git-subtree push -P compiler/rustc_codegen_gcc/ ../rustc_codegen_gcc/ sync_branch_name
cd ../rustc_codegen_gcc
git checkout master
git pull
git checkout sync_branch_name
git merge master
```

To send the changes to the rust repo:

```bash
cd ../rust
git pull origin master
git checkout -b subtree-update_cg_gcc_YYYY-MM-DD
PATH="$HOME/bin:$PATH" ~/bin/git-subtree pull --prefix=compiler/rustc_codegen_gcc/ https://github.com/rust-lang/rustc_codegen_gcc.git master
git push

# Immediately merge the merge commit into cg_gcc to prevent merge conflicts when syncing from rust-lang/rust later.
PATH="$HOME/bin:$PATH" ~/bin/git-subtree push -P compiler/rustc_codegen_gcc/ ../rustc_codegen_gcc/ sync_branch_name
```

TODO: write a script that does the above.

https://rust-lang.zulipchat.com/#narrow/stream/301329-t-devtools/topic/subtree.20madness/near/258877725
