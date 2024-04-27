#!/usr/bin/env bash

wget http://packages.haiku-os.org/haikuports/master/hpkg/llvm-4.0.1-2-x86_64.hpkg
wget http://packages.haiku-os.org/haikuports/master/hpkg/llvm_libs-4.0.1-2-x86_64.hpkg

package extract -C /system llvm-4.0.1-2-x86_64.hpkg
package extract -C /system llvm_libs-4.0.1-2-x86_64.hpkg

rm -f *.hpkg
