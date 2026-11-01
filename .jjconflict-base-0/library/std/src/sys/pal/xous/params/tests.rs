use super::*;
use crate::collections::HashMap;
use crate::io::Write;

fn create_args_test() -> std::io::Result<Vec<u8>> {
    let mut sample_data = vec![];
    let mut h = HashMap::new();

    h.insert("foo", "bar");
    h.insert("baz", "qux");
    h.insert("some", "val");

    // Magic number
    sample_data.write_all(&PARAMS_MAGIC)?;
    // Size of the AppP block
    sample_data.write_all(&4u32.to_le_bytes())?;
    // Number of blocks
    sample_data.write_all(&2u32.to_le_bytes())?;

    // Magic number
    sample_data.write_all(&ENV_MAGIC)?;
    let mut data = vec![];
    for (key, value) in h.iter() {
        data.extend_from_slice(&(key.len() as u16).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&(value.len() as u16).to_le_bytes());
        data.extend_from_slice(value.as_bytes());
    }
    // Size of the EnvB block
    sample_data.write_all(&(data.len() as u32 + 2).to_le_bytes())?;

    // Number of environment variables
    sample_data.write_all(&(h.len() as u16).to_le_bytes())?;

    // Environment variables
    sample_data.write_all(&data)?;

    // Write command line arguments
    let args = vec!["some", "command", "line variable", "entries"];
    sample_data.write_all(&ARGS_MAGIC)?;
    let mut args_size = 0;
    for entry in args.iter() {
        args_size += entry.len() + 2;
    }
    sample_data.write_all(&(args_size as u32 + 2).to_le_bytes())?;
    sample_data.write_all(&(args.len() as u16).to_le_bytes())?;
    for entry in args {
        sample_data.write_all(&(entry.len() as u16).to_le_bytes())?;
        sample_data.write_all(entry.as_bytes())?;
    }

    Ok(sample_data)
}

#[test]
fn basic_arg_parsing() {
    let arg_data = create_args_test().expect("couldn't create test data");
    for byte in &arg_data {
        print!("{:02x} ", byte);
    }
    println!();

    let args = ApplicationParameters::new(&arg_data).expect("Unable to parse arguments");
    for arg in args {
        if let Ok(env) = EnvironmentBlock::try_from(&arg) {
            for env in env {
                println!("{}={}", env.key, env.value);
            }
        } else if let Ok(args) = ArgumentList::try_from(&arg) {
            for arg in args {
                println!("Arg: {}", arg.value);
            }
        }
    }
}
