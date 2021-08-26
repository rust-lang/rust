"""Bootstrap tests"""

from __future__ import absolute_import, division, print_function
import os
import doctest
import unittest
import tempfile
import hashlib
import sys

from shutil import rmtree

import bootstrap


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
        self.rustc_stamp_path = os.path.join(self.container, "stage0",
                                             ".rustc-stamp")
        self.key = self.build.date + str(None)

    def tearDown(self):
        rmtree(self.container)

    def test_stamp_path_does_not_exists(self):
        """Return True when the stamp file does not exists"""
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
        self.assertFalse(self.build.program_out_of_date(self.rustc_stamp_path, self.key))


if __name__ == '__main__':
    SUITE = unittest.TestSuite()
    TEST_LOADER = unittest.TestLoader()
    SUITE.addTest(doctest.DocTestSuite(bootstrap))
    SUITE.addTests([
        TEST_LOADER.loadTestsFromTestCase(VerifyTestCase),
        TEST_LOADER.loadTestsFromTestCase(ProgramOutOfDate)])

    RUNNER = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = RUNNER.run(SUITE)
    sys.exit(0 if result.wasSuccessful() else 1)
