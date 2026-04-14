use clap::Parser as ClapParser;
use kindc::*;
use std::fs;
use std::path::PathBuf;

#[derive(ClapParser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory or single .kind file
    input: PathBuf,

    /// Output directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Parse + validate only, no codegen
    #[arg(long)]
    check: bool,

    /// Print resolved IR as debug output
    #[arg(long)]
    dump_ir: bool,

    /// Verbose diagnostics
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut files = Vec::new();
    if args.input.is_dir() {
        for entry in fs::read_dir(args.input)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("kind") {
                files.push(path);
            }
        }
    } else {
        files.push(args.input.clone());
    }

    if files.is_empty() {
        return Err("No .kind files found".into());
    }

    let mut ast_files = Vec::new();
    for file_path in files {
        let content = fs::read_to_string(&file_path)?;
        let mut parser = Parser::new(file_path.to_string_lossy().to_string(), &content)
            .map_err(|e| format!("Parser init error: {}", e))?;
        let ast = parser.parse_file().map_err(|e| format!("{}: {}", file_path.display(), e))?;
        ast_files.push(ast);
    }

    let mut resolver = Resolver::new();
    let schema = resolver.resolve_files(&ast_files)?;

    validate::validate(&schema)?;

    if args.dump_ir {
        println!("{:#?}", schema);
    }

    if args.check {
        println!("Validation successful.");
        return Ok(());
    }

    let generator = RustGenerator;
    let rust_code = generator.generate(&schema);

    if let Some(out_dir) = args.output {
        fs::create_dir_all(&out_dir)?;
        let out_file = out_dir.join("mod.rs");
        fs::write(out_file, rust_code)?;
    } else {
        println!("{}", rust_code);
    }

    Ok(())
}
