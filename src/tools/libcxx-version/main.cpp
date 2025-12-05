// Detecting the standard library version manually using a bunch of shell commands is very
// complicated and fragile across different platforms. This program provides the major version
// of the standard library on any target platform without requiring any messy work.
//
// It's nothing more than specifying the name of the standard library implementation (either libstdc++ or libc++)
// and its major version.

#include <iostream>

int main() {
    #ifdef _GLIBCXX_RELEASE
        std::cout << "libstdc++ version: " << _GLIBCXX_RELEASE << std::endl;
    #elif defined(_LIBCPP_VERSION)
        // _LIBCPP_VERSION follows "XXYYZZ" format (e.g., 170001 for 17.0.1).
        // ref: https://github.com/llvm/llvm-project/blob/f64732195c1030ee2627ff4e4142038e01df1d26/libcxx/include/__config#L51-L54
        //
        // Since we use the major version from _GLIBCXX_RELEASE, we need to extract only the first 2 characters of _LIBCPP_VERSION
        // to provide the major version for consistency.
        std::cout << "libc++ version: " << std::to_string(_LIBCPP_VERSION).substr(0, 2) << std::endl;
    #else
        std::cerr << "Coudln't recognize the standard library version." << std::endl;
        return 1;
    #endif

    return 0;
}
