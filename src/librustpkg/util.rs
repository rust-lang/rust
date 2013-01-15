pub fn is_cmd(cmd: ~str) -> bool {
    let cmds = &[~"build", ~"clean", ~"install", ~"prefer", ~"test",
                 ~"uninstall", ~"unprefer"];

    vec::contains(cmds, &cmd)
}
