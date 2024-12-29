//! Program to write all inputs from a generator to a file, then invoke a Julia script to plot
//! them. Output is in `target/plots`.
//!
//! Requires Julia with the `CairoMakie` dependency.
//!
//! Note that running in release mode by default generates a _lot_ more datapoints, which
//! causes plotting to be extremely slow (some simplification to be done in the script).

use std::fmt::Write as _;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;
use std::{env, fs};

use libm_test::domain::HasDomain;
use libm_test::gen::{domain_logspace, edge_cases};
use libm_test::{MathOp, op};

const JL_PLOT: &str = "examples/plot_file.jl";

fn main() {
    let manifest_env = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir = Path::new(&manifest_env);
    let out_dir = manifest_dir.join("../../target/plots");
    if !out_dir.exists() {
        fs::create_dir(&out_dir).unwrap();
    }

    let jl_script = manifest_dir.join(JL_PLOT);
    let mut config = format!(r#"out_dir = "{}""#, out_dir.display());
    config.write_str("\n\n").unwrap();

    // Plot a few domains with some functions that use them.
    plot_one_operator::<op::sqrtf::Routine>(&out_dir, &mut config);
    plot_one_operator::<op::cosf::Routine>(&out_dir, &mut config);
    plot_one_operator::<op::cbrtf::Routine>(&out_dir, &mut config);

    let config_path = out_dir.join("config.toml");
    fs::write(&config_path, config).unwrap();

    // The script expects a path to `config.toml` to be passed as its only argument
    let mut cmd = Command::new("julia");
    if cfg!(optimizations_enabled) {
        cmd.arg("-O3");
    }
    cmd.arg(jl_script).arg(config_path);

    println!("launching script... {cmd:?}");
    cmd.status().unwrap();
}

/// Run multiple generators for a single operator.
fn plot_one_operator<Op>(out_dir: &Path, config: &mut String)
where
    Op: MathOp<FTy = f32> + HasDomain<f32>,
{
    plot_one_generator(
        out_dir,
        Op::BASE_NAME.as_str(),
        "logspace",
        config,
        domain_logspace::get_test_cases::<Op>(),
    );
    plot_one_generator(
        out_dir,
        Op::BASE_NAME.as_str(),
        "edge_cases",
        config,
        edge_cases::get_test_cases::<Op, _>(),
    );
}

/// Plot the output of a single generator.
fn plot_one_generator(
    out_dir: &Path,
    fn_name: &str,
    gen_name: &str,
    config: &mut String,
    gen: impl Iterator<Item = (f32,)>,
) {
    let text_file = out_dir.join(format!("input-{fn_name}-{gen_name}.txt"));

    let f = fs::File::create(&text_file).unwrap();
    let mut w = BufWriter::new(f);
    let mut count = 0u64;

    for input in gen {
        writeln!(w, "{:e}", input.0).unwrap();
        count += 1;
    }

    w.flush().unwrap();
    println!("generated {count} inputs for {fn_name}-{gen_name}");

    writeln!(
        config,
        r#"[[input]]
function = "{fn_name}"
generator = "{gen_name}"
input_file = "{}"
"#,
        text_file.to_str().unwrap()
    )
    .unwrap()
}
