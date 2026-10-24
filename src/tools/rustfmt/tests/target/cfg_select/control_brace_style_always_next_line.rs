//@ rustfmt-unstable: true
//@ rustfmt-edition: 2024
//@ rustfmt-control_brace_style: AlwaysNextLine

cfg_select! {
    _ => x,
}

cfg_select! {
    _ => x,
}

cfg_select! {
    _ =>
    {{ x }}
}

cfg_select! {
    _ =>
    {{ x }}
}

cfg_select! {
    _ =>
    unsafe { x },
}

cfg_select! {
    _ =>
    {unsafe { x }}
}
