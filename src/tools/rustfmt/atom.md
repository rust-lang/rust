# Running Rustfmt from Atom

## rust-analyzer

Rustfmt can be utilized from [rust-analyzer](https://rust-analyzer.github.io/) which is provided by [ide-rust](https://atom.io/packages/ide-rust).

`apm install ide-rust`

Once installed a file is formatted with `ctrl-shift-c` or `cmd-shift-c`, also available in context menu.

## atom-beautify

Another way is to install [Beautify](https://atom.io/packages/atom-beautify), you
can do this by running `apm install atom-beautify`.

There are 2 settings that need to be configured in the atom beautifier configuration.

-  Install rustfmt as per the [readme](README.md).
-  Open the atom beautifier settings

   Go to Edit->Preferences. Click the packages on the left side and click on setting for atom-beautifier

-  Set rustfmt as the beautifier

   Find the setting labeled *Language Config - Rust - Default Beautifier* and make sure it is set to rustfmt as shown below. You can also set the beautifier to auto format on save here.
![image](https://cloud.githubusercontent.com/assets/6623285/11147685/c8ade16c-8a3d-11e5-9da5-bd3d998d97f9.png)

-  Set the path to your rustfmt location

   Find the setting labeled *Rust - Rustfmt Path*. This setting is towards the bottom and you will need to scroll a bit. Set it to the path for your rustfmt executable.
![image](https://cloud.githubusercontent.com/assets/6623285/11147718/f4d10224-8a3d-11e5-9f69-9e900cbe0278.png)
