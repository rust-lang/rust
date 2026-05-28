<?xml version="1.0" ?>
<xsl:stylesheet version="1.0"
        xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
        xmlns:wix="http://schemas.microsoft.com/wix/2006/wi">
    <!-- Copy all attributes and elements to the output. -->
    <xsl:template match="@*|*">
        <xsl:copy>
            <xsl:apply-templates select="@*|*"/>
        </xsl:copy>
    </xsl:template>
    <xsl:output method="xml" indent="yes" />

    <!-- Move all files in directory into first component in that directory. -->
    <xsl:template match="wix:Component[1]">
        <xsl:copy>
            <xsl:apply-templates select="@*|*"/>
            <xsl:for-each select="../wix:Component[preceding-sibling::*]/wix:File">
                <xsl:copy>
                    <!-- Component can only have one KeyPath -->
                    <xsl:apply-templates select="@*[not(name()='KeyPath')]|*"/>
                </xsl:copy>
            </xsl:for-each>
        </xsl:copy>
    </xsl:template>

    <!-- Now the rest of components are empty, find them. -->
    <xsl:key name="empty-cmp-ids" match="wix:Component[preceding-sibling::*]" use="@Id" />

    <!-- And remove. -->
    <xsl:template match="wix:Component[preceding-sibling::*]" />

    <!-- Also remove componentsrefs referencing empty components. -->
    <xsl:template match="wix:ComponentRef[key('empty-cmp-ids', @Id)]" />
</xsl:stylesheet>
