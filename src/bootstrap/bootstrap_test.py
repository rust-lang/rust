"""Bootstrap tests

Run these with `x test bootstrap`, or `python -m unittest src/bootstrap/bootstrap_test.py`."""

from __future__ import absolute_import, division, print_function
import os
import unittest
from unittest.mock import patch
import tempfile
import hashlib
import sys

from shutil import rmtree

# Allow running this from the top-level directory.
bootstrap_dir = os.path.dirname(os.path.abspath(__file__))
# For the import below, have Python search in src/bootstrap first.
sys.path.insert(0, bootstrap_dir)
import bootstrap  # noqa: E402
import configure  # noqa: E402


def serialize_and_parse(configure_args, bootstrap_args=None):
    from io import StringIO

    if bootstrap_args is None:
        bootstrap_args = bootstrap.FakeArgs()

    section_order, sections, targets = configure.parse_args(configure_args)
    buffer = StringIO()
    configure.write_config_toml(buffer, section_order, targets, sections)
    build = bootstrap.RustBuild(config_toml=buffer.getvalue(), args=bootstrap_args)

    try:
        import tomllib

        # Verify this is actually valid TOML.
        tomllib.loads(build.config_toml)
    except ImportError:
        print(
            "WARNING: skipping TOML validation, need at least python 3.11",
            file=sys.stderr,
        )
    return build


class VerifyTestCase(unittest.TestCase):
    """Test Case for verify"""

    def setUp(self):
        self.container = tempfile.mkdtemp()
        self.src = os.path.join(self.container, "src.txt")
        self.bad_src = os.path.join(self.container, "bad.txt")
        content = "Hello world"

        self.expected = hashlib.sha256(content.encode("utf-8")).hexdigest()

        with open(self.src, "w") as src:
            src.write(content)
        with open(self.bad_src, "w") as bad:
            bad.write("Hello!")

    def tearDown(self):
        rmtree(self.container)

    def test_valid_file(self):
        """Check if the sha256 sum of the given file is valid"""
        self.assertTrue(bootstrap.verify(self.src, self.expected, False))

    def test_invalid_file(self):
        """Should verify that the file is invalid"""
        self.assertFalse(bootstrap.verify(self.bad_src, self.expected, False))


class ProgramOutOfDate(unittest.TestCase):
    """Test if a program is out of date"""

    def setUp(self):
        self.container = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.container, "stage0"))
        self.build = bootstrap.RustBuild()
        self.build.date = "2017-06-15"
        self.build.build_dir = self.container
        self.rustc_stamp_path = os.path.join(self.container, "stage0", ".rustc-stamp")
        self.key = self.build.date + str(None)

    def tearDown(self):
        rmtree(self.container)

    def test_stamp_path_does_not_exist(self):
        """Return True when the stamp file does not exist"""
        if os.path.exists(self.rustc_stamp_path):
            os.unlink(self.rustc_stamp_path)
        self.assertTrue(self.build.program_out_of_date(self.rustc_stamp_path, self.key))

    def test_dates_are_different(self):
        """Return True when the dates are different"""
        with open(self.rustc_stamp_path, "w") as rustc_stamp:
            rustc_stamp.write("2017-06-14None")
        self.assertTrue(self.build.program_out_of_date(self.rustc_stamp_path, self.key))

    def test_same_dates(self):
        """Return False both dates match"""
        with open(self.rustc_stamp_path, "w") as rustc_stamp:
            rustc_stamp.write("2017-06-15None")
        self.assertFalse(
            self.build.program_out_of_date(self.rustc_stamp_path, self.key)
        )


class ParseArgsInConfigure(unittest.TestCase):
    """Test if `parse_args` function in `configure.py` works properly"""

    @patch("configure.err")
    def test_unknown_args(self, err):
        # It should be print an error message if the argument doesn't start with '--'
        configure.parse_args(["enable-full-tools"])
        err.assert_called_with("Option 'enable-full-tools' is not recognized")
        err.reset_mock()
        # It should be print an error message if the argument is not recognized
        configure.parse_args(["--some-random-flag"])
        err.assert_called_with("Option '--some-random-flag' is not recognized")

    @patch("configure.err")
    def test_need_value_args(self, err):
        """It should print an error message if a required argument value is missing"""
        configure.parse_args(["--target"])
        err.assert_called_with("Option '--target' needs a value (--target=val)")

    @patch("configure.err")
    def test_option_checking(self, err):
        # Options should be checked even if `--enable-option-checking` is not passed
        configure.parse_args(["--target"])
        err.assert_called_with("Option '--target' needs a value (--target=val)")
        err.reset_mock()
        # Options should be checked if `--enable-option-checking` is passed
        configure.parse_args(["--enable-option-checking", "--target"])
        err.assert_called_with("Option '--target' needs a value (--target=val)")
        err.reset_mock()
        # Options should not be checked if `--disable-option-checking` is passed
        configure.parse_args(["--disable-option-checking", "--target"])
        err.assert_not_called()

    @patch("configure.parse_example_config", lambda known_args, _: known_args)
    def test_known_args(self):
        # It should contain known and correct arguments
        known_args = configure.parse_args(["--enable-full-tools"])
        self.assertTrue(known_args["full-tools"][0][1])
        known_args = configure.parse_args(["--disable-full-tools"])
        self.assertFalse(known_args["full-tools"][0][1])
        # It should contain known arguments and their values
        known_args = configure.parse_args(["--target=x86_64-unknown-linux-gnu"])
        self.assertEqual(known_args["target"][0][1], "x86_64-unknown-linux-gnu")
        known_args = configure.parse_args(["--target", "x86_64-unknown-linux-gnu"])
        self.assertEqual(known_args["target"][0][1], "x86_64-unknown-linux-gnu")


class GenerateAndParseConfig(unittest.TestCase):
    """Test that we can serialize and deserialize a bootstrap.toml file"""

    def test_no_args(self):
        build = serialize_and_parse([])
        self.assertEqual(build.get_toml("profile"), "dist")
        self.assertIsNone(build.get_toml("llvm.download-ci-llvm"))

    def test_set_section(self):
        build = serialize_and_parse(["--set", "llvm.download-ci-llvm"])
        self.assertEqual(build.get_toml("download-ci-llvm", section="llvm"), "true")

    def test_set_target(self):
        build = serialize_and_parse(["--set", "target.x86_64-unknown-linux-gnu.cc=gcc"])
        self.assertEqual(
            build.get_toml("cc", section="target.x86_64-unknown-linux-gnu"), "gcc"
        )

    def test_set_top_level(self):
        build = serialize_and_parse(["--set", "profile=compiler"])
        self.assertEqual(build.get_toml("profile"), "compiler")

    def test_set_codegen_backends(self):
        build = serialize_and_parse(["--set", "rust.codegen-backends=cranelift"])
        self.assertNotEqual(
            build.config_toml.find("codegen-backends = ['cranelift']"), -1
        )
        build = serialize_and_parse(["--set", "rust.codegen-backends=cranelift,llvm"])
        self.assertNotEqual(
            build.config_toml.find("codegen-backends = ['cranelift', 'llvm']"), -1
        )
        build = serialize_and_parse(["--enable-full-tools"])
        self.assertNotEqual(build.config_toml.find("codegen-backends = ['llvm']"), -1)


class BuildBootstrap(unittest.TestCase):
    """Test that we generate the appropriate arguments when building bootstrap"""

    def build_args(self, configure_args=None, args=None, env=None):
        if configure_args is None:
            configure_args = []
        if args is None:
            args = []
        if env is None:
            env = {}

        # This test ends up invoking build_bootstrap_cmd, which searches for
        # the Cargo binary and errors out if it cannot be found. This is not a
        # problem in most cases, but there is a scenario where it would cause
        # the test to fail.
        #
        # When a custom local Cargo is configured in bootstrap.toml (with the
        # build.cargo setting), no Cargo is downloaded to any location known by
        # bootstrap, and bootstrap relies on that setting to find it.
        #
        # In this test though we are not using the bootstrap.toml of the caller:
        # we are generating a blank one instead. If we don't set build.cargo in
        # it, the test will have no way to find Cargo, failing the test.
        cargo_bin = os.environ.get("BOOTSTRAP_TEST_CARGO_BIN")
        if cargo_bin is not None:
            configure_args += ["--set", "build.cargo=" + cargo_bin]
        rustc_bin = os.environ.get("BOOTSTRAP_TEST_RUSTC_BIN")
        if rustc_bin is not None:
            configure_args += ["--set", "build.rustc=" + rustc_bin]

        env = env.copy()
        env["PATH"] = os.environ["PATH"]

        parsed = bootstrap.parse_args(args)
        build = serialize_and_parse(configure_args, parsed)
        # Make these optional so that `python -m unittest` works when run manually.
        build_dir = os.environ.get("BUILD_DIR")
        if build_dir is not None:
            build.build_dir = build_dir
        build_platform = os.environ.get("BUILD_PLATFORM")
        if build_platform is not None:
            build.build = build_platform
        return build.build_bootstrap_cmd(env), env

    def test_cargoflags(self):
        args, _ = self.build_args(env={"CARGOFLAGS": "--timings"})
        self.assertTrue("--timings" in args)

    def test_warnings(self):
        for toml_warnings in ["false", "true", None]:
            configure_args = []
            if toml_warnings is not None:
                configure_args = ["--set", "rust.deny-warnings=" + toml_warnings]

            _, env = self.build_args(configure_args, args=["--warnings=warn"])
            self.assertFalse("-Dwarnings" in env["RUSTFLAGS"])

            _, env = self.build_args(configure_args, args=["--warnings=deny"])
            self.assertTrue("-Dwarnings" in env["RUSTFLAGS"])
