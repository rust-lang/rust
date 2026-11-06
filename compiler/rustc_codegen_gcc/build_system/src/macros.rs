macro_rules! arg_error {
    ($($err:tt)*) => {{
        eprintln!($($err)*);
        eprintln!();
        usage();
        std::process::exit(1);
    }};
}

macro_rules! sub_command_decl {
    (__inner, $doc_name:literal, $instance:ident, $args:ident, $field:ident, bool) => {{
        $instance.$field = true;
    }};
    (__inner, $doc_name:literal, $instance:ident, $args:ident, $field:ident, Option<String>) => {{
        if let Some(value) = $args.next() {
            $instance.$field = Some(value);
        } else {
            return Err(
                concat!("Expected a value after `", $doc_name, "`, found nothing").to_string()
            );
        }
    }};
    (__inner, $doc_name:literal, $instance:ident, $args:ident, $field:ident, PathBuf) => {{
        match $args.next() {
            Some(path) if !path.is_empty() => $instance.$field = path.into(),
            _ => {
                return Err(
                    concat!("Expected an argument after `", $doc_name, "`, found nothing").into()
                );
            }
        }
    }};
    (__arm, $instance:ident, $arg:ident, $args:ident, $extra:ident) => {{
        if !$instance.$extra.parse_argument($arg, &mut $args)? {
            return Err(format!(
                "Unknown argument {:?}. Use `--help` to see the list of available options.", $arg
            ))
        }
    }};
    (__arm, $instance:ident, $arg:ident, $args:ident) => {{
        return Err(format!(
            "Unknown argument {:?}. Use `--help` to see the list of available options.", $arg
        ))
    }};
    (
        $name:ident => $command_name:literal,
        $($field:ident: $typ:ident $(<$generic:ident>)? = $doc_name:literal => $doc:literal,)+
        $(+ $extra:ident: $extra_ty:ident)?
        $(~ $default_field:ident = $default_value:expr,)*
    ) => {
        #[derive(Default)]
        struct $name {
            $($field: $typ $(<$generic>)?,)+
            $($extra: $extra_ty,)?
        }

        impl $name {
            fn new() -> Result<Option<Self>, String> {
                let mut instance = Self::default();
                $(instance.$default_field = $default_value;)*
                let mut args = std::env::args().skip(2);
                while let Some(arg) = args.next() {
                    match arg.as_str() {
                        $($doc_name => sub_command_decl!(
                            __inner, $doc_name, instance, args, $field, $typ $(<$generic>)?),)+
                        "--help" => {
                            Self::usage();
                            return Ok(None);
                        }
                        arg => sub_command_decl!(__arm, instance, arg, args $(, $extra)?),
                    }
                }
                Ok(Some(instance))
            }

            fn usage() {
                println!(concat!("`", $command_name, "` command help:\n"));

                let mut commands = vec![$(($doc_name, $doc),)+];
                commands.sort_unstable_by(|a, b| a.0.cmp(b.0));
                let longest = commands.iter().map(|(name, _)| name.len()).max().unwrap();
                for (name, doc) in commands {
                    let spacing = std::iter::repeat(' ').take(longest - name.len() + 1).collect::<String>();
                    println!("    {name}{spacing}: {doc}");
                }
                $($extra_ty::show_usage();)?
                println!("    --help : Show this help");
            }
        }
    }
}

macro_rules! commands_decl {
    ($($variant:ident: $doc_name:literal => $doc:literal ,)+) => {
        enum Command {
            $($variant),+
        }

        impl<'a> From<Option<&'a str>> for Command {
            fn from(arg: Option<&'a str>) -> Self {
                match arg {
                    $(Some($doc_name) => Self::$variant,)+
                    Some("--help") => {
                        usage();
                        std::process::exit(0);
                    }
                    Some(flag) if flag.starts_with('-') => arg_error!("Expected command found flag {}", flag),
                    Some(command) => arg_error!("Unknown command {}", command),
                    None => {
                        usage();
                        std::process::exit(0);
                    }
                }
            }
        }

        fn usage() {
            println!("\
rustc_codegen_gcc build system

Usage: build_system [command] [options]

Options:
    --help     : Displays this help message.

Commands:",
            );
            let mut commands = vec![$(($doc_name, $doc),)+];
            let longest = commands.iter().map(|(name, _)| name.len()).max().unwrap();

            commands.sort_unstable_by(|a, b| a.0.cmp(b.0));
            for (name, doc) in commands {
                let spacing = std::iter::repeat(' ').take(longest - name.len() + 1).collect::<String>();
                println!("    {name}{spacing}: {doc}.");
            }
        }
    }
}
