# Running Rustfmt from IntelliJ or CLion

## Installation

- Install [CLion](https://www.jetbrains.com/clion/), [IntelliJ Ultimate or CE](https://www.jetbrains.com/idea/) through the direct download link or using the [JetBrains Toolbox](https://www.jetbrains.com/toolbox/).
  CLion provides a built-in debugger interface but its not free like IntelliJ CE - which does not provide the debugger interface. (IntelliJ seems to lack the toolchain for that, see this discussion [intellij-rust/issues/535](https://github.com/intellij-rust/intellij-rust/issues/535))
  
- Install the [Rust Plugin](https://intellij-rust.github.io/) by navigating to File -> Settings -> Plugins and press "Install JetBrains Plugin"
  ![plugins](https://user-images.githubusercontent.com/1133787/47240861-f40af680-d3e9-11e8-9b82-cdd5c8d5f5b8.png)

- Press "Install" on the rust plugin
  ![install rust](https://user-images.githubusercontent.com/1133787/47240803-c0c86780-d3e9-11e8-9265-22f735e4d7ed.png)
  
- Restart CLion/IntelliJ

## Configuration

- Open the settings window (File -> Settings) and search for "reformat"
  ![keymap](https://user-images.githubusercontent.com/1133787/47240922-2ae10c80-d3ea-11e8-9d8f-c798d9749240.png)
- Right-click on "Reformat File with Rustfmt" and assign a keyboard shortcut

  ![shortcut_window](https://user-images.githubusercontent.com/1133787/47240981-5b28ab00-d3ea-11e8-882e-8b864164db74.png)
- Press "OK"
  ![shortcut_after](https://user-images.githubusercontent.com/1133787/47241000-6976c700-d3ea-11e8-9342-50ebc2f9f97b.png)
  
- Done. You can now use rustfmt in an opened *.rs file with your previously specified shortcut
