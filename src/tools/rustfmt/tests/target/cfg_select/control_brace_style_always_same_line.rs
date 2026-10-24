//@ rustfmt-unstable: true
//@ rustfmt-control_brace_style: AlwaysSameLine
//                               ^ NB: default value

cfg_select! {
    _ => x,
}

cfg_select! {
    _ => x,
}

cfg_select! {
    _ => {{ x }}
}

cfg_select! {
    _ => {{ x }}
}

cfg_select! {
    _ => unsafe { x },
}

cfg_select! {
    _ => {unsafe { x }}
}
