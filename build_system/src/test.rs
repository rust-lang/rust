use crate::utils::run_command_with_output;

fn get_args<'a>(args: &mut Vec<&'a dyn AsRef<std::ffi::OsStr>>, extra_args: &'a Vec<String>) {
    for extra_arg in extra_args {
        args.push(extra_arg);
    }
}

pub fn run() -> Result<(), String> {
    let mut args: Vec<&dyn AsRef<std::ffi::OsStr>> = vec![&"bash", &"test.sh"];
    let extra_args = std::env::args().skip(2).collect::<Vec<_>>();
    get_args(&mut args, &extra_args);
    let current_dir = std::env::current_dir().map_err(|error| format!("`current_dir` failed: {:?}", error))?;
    run_command_with_output(args.as_slice(), Some(&current_dir))
}
