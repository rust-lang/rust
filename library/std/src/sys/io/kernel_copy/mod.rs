cfg_select! {
    any(target_os = "linux", target_os = "android") => {
        mod linux;
    }
    _ => { }
}
