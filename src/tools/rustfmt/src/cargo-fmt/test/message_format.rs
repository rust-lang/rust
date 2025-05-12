use super::*;

#[test]
fn invalid_message_format() {
    assert_eq!(
        convert_message_format_to_rustfmt_args("awesome", &mut vec![]),
        Err(String::from(
            "invalid --message-format value: awesome. Allowed values are: short|json|human"
        )),
    );
}

#[test]
fn json_message_format_and_check_arg() {
    let mut args = vec![String::from("--check")];
    assert_eq!(
        convert_message_format_to_rustfmt_args("json", &mut args),
        Err(String::from(
            "cannot include --check arg when --message-format is set to json"
        )),
    );
}

#[test]
fn json_message_format_and_emit_arg() {
    let mut args = vec![String::from("--emit"), String::from("checkstyle")];
    assert_eq!(
        convert_message_format_to_rustfmt_args("json", &mut args),
        Err(String::from(
            "cannot include --emit arg when --message-format is set to json"
        )),
    );
}

#[test]
fn json_message_format() {
    let mut args = vec![String::from("--edition"), String::from("2018")];
    assert!(convert_message_format_to_rustfmt_args("json", &mut args).is_ok());
    assert_eq!(
        args,
        vec![
            String::from("--edition"),
            String::from("2018"),
            String::from("--emit"),
            String::from("json")
        ]
    );
}

#[test]
fn human_message_format() {
    let exp_args = vec![String::from("--emit"), String::from("json")];
    let mut act_args = exp_args.clone();
    assert!(convert_message_format_to_rustfmt_args("human", &mut act_args).is_ok());
    assert_eq!(act_args, exp_args);
}

#[test]
fn short_message_format() {
    let mut args = vec![String::from("--check")];
    assert!(convert_message_format_to_rustfmt_args("short", &mut args).is_ok());
    assert_eq!(args, vec![String::from("--check"), String::from("-l")]);
}

#[test]
fn short_message_format_included_short_list_files_flag() {
    let mut args = vec![String::from("--check"), String::from("-l")];
    assert!(convert_message_format_to_rustfmt_args("short", &mut args).is_ok());
    assert_eq!(args, vec![String::from("--check"), String::from("-l")]);
}

#[test]
fn short_message_format_included_long_list_files_flag() {
    let mut args = vec![String::from("--check"), String::from("--files-with-diff")];
    assert!(convert_message_format_to_rustfmt_args("short", &mut args).is_ok());
    assert_eq!(
        args,
        vec![String::from("--check"), String::from("--files-with-diff")]
    );
}
