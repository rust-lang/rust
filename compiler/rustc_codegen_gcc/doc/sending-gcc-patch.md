This guide explains what to do to send a GCC patch for review.

All the commands are supposed to be run in the folder where you cloned GCC.

```bash
./contrib/gcc-changelog/git_check_commit.py
```

You can provide a specific commit hash:

```bash
./contrib/gcc-changelog/git_check_commit.py abdef78989
```

a range:

```bash
./contrib/gcc-changelog/git_check_commit.py HEAD~2
```

or even a comparison with a remote branch:

```bash
./contrib/gcc-changelog/git_check_commit.py upstream/master..HEAD
```

When there is no more errors, generate the git patch:

```bash
git format-patch -1 `git rev-parse --short HEAD`
```

Then you can run the remaining checks using:

```bash
contrib/check_GNU_style.sh 0001-your-patch.patch
```

When you have no more errors, you can send the `.patch` file to GCC by sending an
email to `gcc-patches@gcc.gnu.org` and to the relevant GCC mailing lists
depending on what your patch changes. You can find the list of the mailing lists
[here](https://gcc.gnu.org/lists.html).

You can find more information about "contributing to GCC" [here](https://gcc.gnu.org/contribute.html).
