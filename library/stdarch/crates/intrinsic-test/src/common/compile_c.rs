#[derive(Clone)]
pub struct CompilationCommandBuilder {
    compiler: String,
    target: Option<String>,
    cxx_toolchain_dir: Option<String>,
    arch_flags: Vec<String>,
    optimization: String,
    project_root: Option<String>,
    extra_flags: Vec<String>,
}

impl CompilationCommandBuilder {
    pub fn new() -> Self {
        Self {
            compiler: String::new(),
            target: None,
            cxx_toolchain_dir: None,
            arch_flags: Vec::new(),
            optimization: "2".to_string(),
            project_root: None,
            extra_flags: Vec::new(),
        }
    }

    pub fn set_compiler(mut self, compiler: &str) -> Self {
        self.compiler = compiler.to_string();
        self
    }

    pub fn set_target(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }

    pub fn set_cxx_toolchain_dir(mut self, path: Option<&str>) -> Self {
        self.cxx_toolchain_dir = path.map(|p| p.to_string());
        self
    }

    pub fn add_arch_flags<'a>(mut self, flags: impl IntoIterator<Item = &'a str>) -> Self {
        self.arch_flags
            .extend(flags.into_iter().map(|s| s.to_owned()));

        self
    }

    pub fn set_opt_level(mut self, optimization: &str) -> Self {
        self.optimization = optimization.to_string();
        self
    }

    /// Sets the root path of all the generated test files.
    pub fn set_project_root(mut self, path: &str) -> Self {
        self.project_root = Some(path.to_string());
        self
    }

    pub fn add_extra_flags<'a>(mut self, flags: impl IntoIterator<Item = &'a str>) -> Self {
        self.extra_flags
            .extend(flags.into_iter().map(|s| s.to_owned()));

        self
    }

    pub fn add_extra_flag(self, flag: &str) -> Self {
        self.add_extra_flags([flag])
    }
}

impl CompilationCommandBuilder {
    pub fn into_cpp_compilation(self) -> CppCompilation {
        let mut cpp_compiler = std::process::Command::new(self.compiler);

        if let Some(project_root) = self.project_root {
            cpp_compiler.current_dir(project_root);
        }

        let flags = std::env::var("CPPFLAGS").unwrap_or("".into());
        cpp_compiler.args(flags.split_whitespace());

        cpp_compiler.arg(format!("-march={}", self.arch_flags.join("+")));

        cpp_compiler.arg(format!("-O{}", self.optimization));

        cpp_compiler.args(self.extra_flags);

        if let Some(target) = &self.target {
            cpp_compiler.arg(format!("--target={target}"));
        }

        CppCompilation(cpp_compiler)
    }
}

pub struct CppCompilation(std::process::Command);

fn clone_command(command: &std::process::Command) -> std::process::Command {
    let mut cmd = std::process::Command::new(command.get_program());
    if let Some(current_dir) = command.get_current_dir() {
        cmd.current_dir(current_dir);
    }
    cmd.args(command.get_args());

    for (key, val) in command.get_envs() {
        cmd.env(key, val.unwrap_or_default());
    }

    cmd
}

impl CppCompilation {
    pub fn command_mut(&mut self) -> &mut std::process::Command {
        &mut self.0
    }

    pub fn compile_object_file(
        &self,
        input: &str,
        output: &str,
    ) -> std::io::Result<std::process::Output> {
        let mut cmd = clone_command(&self.0);
        cmd.args([input, "-c", "-o", output]);
        cmd.output()
    }

    pub fn link_executable(
        &self,
        inputs: impl Iterator<Item = String>,
        output: &str,
    ) -> std::io::Result<std::process::Output> {
        let mut cmd = clone_command(&self.0);
        cmd.args(inputs);
        cmd.args(["-o", output]);
        cmd.output()
    }
}
