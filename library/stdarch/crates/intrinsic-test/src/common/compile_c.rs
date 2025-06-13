#[derive(Clone)]
pub struct CompilationCommandBuilder {
    compiler: String,
    target: Option<String>,
    cxx_toolchain_dir: Option<String>,
    arch_flags: Vec<String>,
    optimization: String,
    include_paths: Vec<String>,
    project_root: Option<String>,
    output: String,
    input: String,
    linker: Option<String>,
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
            include_paths: Vec::new(),
            project_root: None,
            output: String::new(),
            input: String::new(),
            linker: None,
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

    pub fn add_arch_flags(mut self, flags: Vec<&str>) -> Self {
        let mut new_arch_flags = flags.into_iter().map(|v| v.to_string()).collect();
        self.arch_flags.append(&mut new_arch_flags);

        self
    }

    pub fn set_opt_level(mut self, optimization: &str) -> Self {
        self.optimization = optimization.to_string();
        self
    }

    /// Sets a list of include paths for compilation.
    /// The paths that are passed must be relative to the
    /// "cxx_toolchain_dir" directory path.
    pub fn set_include_paths(mut self, paths: Vec<&str>) -> Self {
        self.include_paths = paths.into_iter().map(|path| path.to_string()).collect();
        self
    }

    /// Sets the root path of all the generated test files.
    pub fn set_project_root(mut self, path: &str) -> Self {
        self.project_root = Some(path.to_string());
        self
    }

    /// The name of the output executable, without any suffixes
    pub fn set_output_name(mut self, path: &str) -> Self {
        self.output = path.to_string();
        self
    }

    /// The name of the input C file, without any suffixes
    pub fn set_input_name(mut self, path: &str) -> Self {
        self.input = path.to_string();
        self
    }

    pub fn set_linker(mut self, linker: String) -> Self {
        self.linker = Some(linker);
        self
    }

    pub fn add_extra_flags(mut self, flags: Vec<&str>) -> Self {
        let mut flags: Vec<String> = flags.into_iter().map(|f| f.to_string()).collect();
        self.extra_flags.append(&mut flags);
        self
    }

    pub fn add_extra_flag(self, flag: &str) -> Self {
        self.add_extra_flags(vec![flag])
    }
}

impl CompilationCommandBuilder {
    pub fn make_string(self) -> String {
        let arch_flags = self.arch_flags.join("+");
        let flags = std::env::var("CPPFLAGS").unwrap_or("".into());
        let project_root = self.project_root.unwrap_or_default();
        let project_root_str = project_root.as_str();
        let mut output = self.output.clone();
        if self.linker.is_some() {
            output += ".o"
        };
        let mut command = format!(
            "{} {flags} -march={arch_flags} \
            -O{} \
            -o {project_root}/{} \
            {project_root}/{}.cpp",
            self.compiler, self.optimization, output, self.input,
        );

        command = command + " " + self.extra_flags.join(" ").as_str();

        if let Some(target) = &self.target {
            command = command + " --target=" + target;
        }

        if let (Some(linker), Some(cxx_toolchain_dir)) = (&self.linker, &self.cxx_toolchain_dir) {
            let include_args = self
                .include_paths
                .iter()
                .map(|path| "--include-directory=".to_string() + cxx_toolchain_dir + path)
                .collect::<Vec<_>>()
                .join(" ");

            command = command
                + " -c "
                + include_args.as_str()
                + " && "
                + linker
                + " "
                + project_root_str
                + "/"
                + &output
                + " -o "
                + project_root_str
                + "/"
                + &self.output
                + " && rm "
                + project_root_str
                + "/"
                + &output;
        }
        command
    }
}
