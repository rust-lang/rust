// check-pass

fn main() {
    let _ = option_env!("PATH").unwrap();
    //~^ WARN incorrect usage of `option_env!`
    let _ = option_env!("PATH").expect("environment variable PATH isn't set");
    //~^ WARN incorrect usage of `option_env!`
    let _ = option_env!("NOT_IN_ENV").unwrap();
    //~^ WARN incorrect usage of `option_env!`
    let _ = option_env!("NOT_IN_ENV").expect("environment variable NOT_IN_ENV isn't set");
    //~^ WARN incorrect usage of `option_env!`
    let _ = assert_ne!(option_env!("PATH").unwrap(), "a");
    //~^ WARN incorrect usage of `option_env!`

    // just verify suggestion
    let _ = option_env!("PATH")
    //~^ WARN incorrect usage of `option_env!`
        .unwrap();
    let _ = option_env!(
    //~^ WARN incorrect usage of `option_env!`
            "PATH"
        )
        . unwrap();
    let _ = (option_env!("NOT_IN_ENV")).expect("aaa");
    //~^ WARN incorrect usage of `option_env!`

    // should not lint
    let _ = option_env!("PATH").unwrap_or("...");
    let _ = option_env!("PATH").unwrap_or_else(|| "...");
}
