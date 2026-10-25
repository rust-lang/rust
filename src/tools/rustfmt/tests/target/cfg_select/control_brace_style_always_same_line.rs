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

cfg_select! {
    _ =>
    // line comment
    {{ x }}
}

cfg_select! {
    _ =>
    // line comment
    {
        a;
        b
    }
}

cfg_select! {
    _ =>
    /* after */
    {{ x }}
}

cfg_select! {
    _ =>
    /* after */
    {
        a;
        b
    }
}

cfg_select! {
    _ =>
    /* after
    multiline */
    {{ x }}
}

cfg_select! {
    _ =>
    /* after
    multiline */
    {
        a;
        b
    }
}

cfg_select! {
    _ =>
    /* after */
    {{ x }}
}

cfg_select! {
    _ =>
    /* after */
    {
        a;
        b
    }
}
